"""
生成 DPO 偏好数据
=================
用 SFT 模型对 MBPP 题目生成多个 completion，
执行测试自动标注 chosen/rejected

用法:
  python generate_dpo_data.py --model outputs/sft-merged
  python generate_dpo_data.py --model outputs/sft-merged --backend hf
"""

import json
import os
import re
import sys
import subprocess
import tempfile
import time
import argparse
from datasets import load_dataset, concatenate_datasets

NUM_COMPLETIONS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 1024


def infer_function_name(test_list):
    for test in test_list:
        match = re.search(r'assert\s+(\w+)\s*\(', test)
        if match:
            return match.group(1)
    return ""


def extract_code(response):
    """提取代码"""
    m = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "```python" in response:
        return response.split("```python")[1].split("```")[0].strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned and ('def ' in cleaned or 'return ' in cleaned):
        return cleaned
    return response.strip()


def execute_with_tests(code, test_list, test_setup_code=""):
    """用 subprocess 执行，5 秒超时"""
    full_code = ""
    if test_setup_code:
        full_code += test_setup_code + "\n"
    full_code += code + "\n"

    num_total = len(test_list)
    num_passed = 0

    for test in test_list:
        exec_code = full_code + "\n" + test
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(exec_code)
                tmp = f.name
            result = subprocess.run(
                [sys.executable, tmp],
                capture_output=True, text=True, timeout=5
            )
            os.unlink(tmp)
            if result.returncode == 0:
                num_passed += 1
        except subprocess.TimeoutExpired:
            try:
                os.unlink(tmp)
            except:
                pass
        except:
            pass

    return {"passed": num_passed == num_total, "num_passed": num_passed, "num_total": num_total}


def build_prompts(dataset):
    prompts = []
    for item in dataset:
        func_name = infer_function_name(item["test_list"])
        user_content = item["text"]
        if func_name:
            user_content += f"\n\nThe function should be named `{func_name}`."
        if item["test_list"]:
            user_content += f"\n\nExample test case:\n{item['test_list'][0]}"

        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Complete the following function. Return ONLY the complete function implementation."},
            {"role": "user", "content": user_content}
        ]
        prompts.append({
            "task_id": item["task_id"],
            "messages": messages,
            "test_list": item["test_list"],
            "test_setup_code": item.get("test_setup_code", ""),
        })
    return prompts


def generate_vllm(prompts, model_path, num_completions):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\nLoading vLLM: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True)
    params = SamplingParams(n=num_completions, temperature=TEMPERATURE, top_p=0.95, max_tokens=MAX_TOKENS, stop=["<|endoftext|>", "<|im_end|>"])

    texts = [tokenizer.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True) for p in prompts]
    print(f"  {len(texts)} prompts x {num_completions} completions")

    t0 = time.time()
    outputs = llm.generate(texts, params)
    print(f"  Done in {time.time()-t0:.1f}s")

    results = []
    for p, out in zip(prompts, outputs):
        results.append({**p, "completions": [o.text for o in out.outputs]})
    return results


def generate_hf(prompts, model_path, num_completions):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\nLoading HF: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    results = []
    for i, p in enumerate(prompts):
        if i % 50 == 0:
            print(f"  {i}/{len(prompts)}")
        text = tokenizer.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        completions = []
        for _ in range(num_completions):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            completions.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        results.append({**p, "completions": completions})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--output-dir", default="outputs/dpo-data")
    parser.add_argument("--num-completions", type=int, default=NUM_COMPLETIONS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MBPP
    print("Loading MBPP...")
    ds = load_dataset("google-research-datasets/mbpp", "full")
    data = concatenate_datasets([ds["train"], ds["validation"]])
    print(f"  {len(data)} problems")

    prompts = build_prompts(data)

    # Generate
    if args.backend == "vllm":
        results = generate_vllm(prompts, args.model, args.num_completions)
    else:
        results = generate_hf(prompts, args.model, args.num_completions)

    # Label
    print("\nLabeling with tests...")
    total_pass = 0
    total_fail = 0
    labeled = []

    for i, item in enumerate(results):
        if i % 50 == 0:
            print(f"  {i}/{len(results)}")
        comps = []
        for resp in item["completions"]:
            code = extract_code(resp)
            result = execute_with_tests(code, item["test_list"], item.get("test_setup_code", ""))
            comps.append({"response": resp, "test_result": result})
            if result["passed"]:
                total_pass += 1
            else:
                total_fail += 1
        labeled.append({**item, "labeled_completions": comps})

    total = total_pass + total_fail
    print(f"\n  Pass: {total_pass}/{total} ({100*total_pass/max(total,1):.1f}%)")

    # Build DPO pairs
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dpo_pairs = []
    for item in labeled:
        passed = [c for c in item["labeled_completions"] if c["test_result"]["passed"]]
        failed = [c for c in item["labeled_completions"] if not c["test_result"]["passed"]]
        if not passed or not failed:
            continue

        prompt_str = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
        count = 0
        for ch in passed:
            for rj in failed:
                if count >= 3:
                    break
                dpo_pairs.append({"prompt": prompt_str, "chosen": ch["response"], "rejected": rj["response"]})
                count += 1
            if count >= 3:
                break

    dpo_path = os.path.join(args.output_dir, "dpo_pairs.json")
    with open(dpo_path, "w") as f:
        json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"DPO pairs: {len(dpo_pairs)}")
    print(f"Saved to: {dpo_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
