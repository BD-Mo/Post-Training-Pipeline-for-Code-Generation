import json
import re
import sys
from human_eval.data import read_problems


def extract_body(response, entry_point):
    # clean markdown
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            response = parts[1]
    # clean <code> tags
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

    # no def found, clean up and return
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--output", default="outputs/humaneval_dpo_results.jsonl")
    args = parser.parse_args()

    problems = read_problems()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model_path, max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True)
        params = SamplingParams(max_tokens=512, temperature=0, stop=["<|endoftext|>", "<|im_end|>"])
        outputs = llm.generate(prompts_list, params)
        raw_responses = [o.outputs[0].text for o in outputs]
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        raw_responses = []
        for i, text in enumerate(prompts_list):
            if i % 20 == 0:
                print(f"  {i}/{len(prompts_list)}")
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw_responses.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    samples = []
    for task_id, ep, resp in zip(task_ids, entry_points, raw_responses):
        body = extract_body(resp, ep)
        samples.append({"task_id": task_id, "completion": body})

    print("\n=== Preview: first 3 samples ===")
    for s in samples[:3]:
        print(f"\n{s['task_id']}:")
        print(s["completion"][:200])

    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
