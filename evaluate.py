"""
统一评估脚本 - 所有阶段都用这一个
==========================================
用法:
  # Pass@1 (greedy)
  python evaluate.py --model Qwen/Qwen2.5-Coder-3B-Instruct --output results/base_p1.jsonl

  # Pass@1,5,10 (sampling)
  python evaluate.py --model outputs/sft-merged --n 10 --temp 0.8 --output results/sft_p10.jsonl

  # 只评估已有的 jsonl
  python evaluate.py --eval-only --input results/base_p1.jsonl --k 1
  python evaluate.py --eval-only --input results/sft_p10.jsonl --k 1 5 10
"""

import json
import re
import argparse
from human_eval.data import read_problems


def extract_body(response, entry_point):
    """从模型输出中提取函数体"""
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            response = parts[1]

    m = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
    if m:
        response = m.group(1)

    lines = response.strip().split("\n")

    # find def entry_point(
    start_idx = -1
    for i, line in enumerate(lines):
        if re.match(r"\s*def\s+" + re.escape(entry_point) + r"\s*\(", line):
            start_idx = i
            break

    if start_idx >= 0:
        remaining = lines[start_idx + 1:]
        body_lines = []
        in_ds = False
        done_ds = False
        for line in remaining:
            s = line.strip()
            if not done_ds:
                if s == "":
                    continue
                if s.startswith('"""') or s.startswith("'''"):
                    q = s[:3]
                    if s.count(q) >= 2 and len(s) > 3:
                        done_ds = True
                        continue
                    else:
                        in_ds = True
                        continue
                if in_ds:
                    if '"""' in s or "'''" in s:
                        in_ds = False
                        done_ds = True
                    continue
                done_ds = True
            body_lines.append(line)
        result = "\n".join(body_lines)
        if result.strip():
            return result + "\n"

    # no def found, clean up
    clean = []
    for line in lines:
        s = line.strip()
        if s.startswith("from ") or s.startswith("import "):
            continue
        if re.match(r"\s*def\s+", s):
            continue
        clean.append(line)
    result = "\n".join(clean).strip()
    if not result:
        return "    pass\n"
    if not result.startswith(" ") and not result.startswith("\t"):
        result = "    " + result.replace("\n", "\n    ")
    return result + "\n"


def generate_and_save(model_path, output_path, n, temperature, backend):
    """用 vLLM 或 HF 生成 completions 并保存"""
    problems = read_problems()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts_list = []
    task_ids = []
    entry_points = []
    for task_id, p in problems.items():
        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Complete the following function. Return ONLY the complete function implementation."},
            {"role": "user", "content": p["prompt"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_list.append(text)
        task_ids.append(task_id)
        entry_points.append(p["entry_point"])

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_path, max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True)
        if temperature == 0:
            params = SamplingParams(n=1, max_tokens=512, temperature=0, stop=["<|endoftext|>", "<|im_end|>"])
        else:
            params = SamplingParams(n=n, max_tokens=512, temperature=temperature, top_p=0.95, stop=["<|endoftext|>", "<|im_end|>"])
        outputs = llm.generate(prompts_list, params)
        samples = []
        for task_id, ep, output in zip(task_ids, entry_points, outputs):
            for comp in output.outputs:
                body = extract_body(comp.text, ep)
                samples.append({"task_id": task_id, "completion": body})
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
        model.eval()
        samples = []
        for i, (text, task_id, ep) in enumerate(zip(prompts_list, task_ids, entry_points)):
            if i % 20 == 0:
                print(f"  {i}/{len(prompts_list)}")
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            actual_n = 1 if temperature == 0 else n
            for _ in range(actual_n):
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=512,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else None,
                        top_p=0.95 if temperature > 0 else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                body = extract_body(resp, ep)
                samples.append({"task_id": task_id, "completion": body})

    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # preview
    print(f"\n=== Preview ===")
    for s in samples[:2]:
        print(f"\n{s['task_id']}:")
        print(s["completion"][:150])
    print(f"\nSaved {len(samples)} completions to {output_path}")


def evaluate(input_path, k_values):
    """评估 Pass@k"""
    from human_eval.evaluation import evaluate_functional_correctness
    results = evaluate_functional_correctness(input_path, k=k_values, n_workers=4, timeout=10.0)
    print("\n" + "=" * 40)
    print("HumanEval Results")
    print("=" * 40)
    for key, val in sorted(results.items()):
        print(f"  {key}: {val*100:.1f}%")
    print("=" * 40)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--output", type=str, default="results/output.jsonl")
    parser.add_argument("--n", type=int, default=1, help="Completions per problem")
    parser.add_argument("--temp", type=float, default=0.0, help="0=greedy, >0=sampling")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing jsonl")
    parser.add_argument("--input", type=str, help="Input jsonl for eval-only mode")
    parser.add_argument("--k", type=int, nargs="+", default=[1], help="k values for Pass@k")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    if args.eval_only:
        path = args.input or args.output
        evaluate(path, args.k)
    else:
        if not args.model:
            parser.error("--model is required when not using --eval-only")
        generate_and_save(args.model, args.output, args.n, args.temp, args.backend)
        evaluate(args.output, args.k)


if __name__ == "__main__":
    main()
