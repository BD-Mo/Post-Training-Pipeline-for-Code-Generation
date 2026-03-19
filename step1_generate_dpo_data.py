"""
Step 1 (修复版): 从 SFT 模型生成 DPO 训练数据
==============================================
修复:
  1. 代码提取支持 SFT 的 <code>...</code> 格式
  2. Prompt 中加入函数签名提示（从 test_list 推断）
  3. 可复用已有的 raw_completions（不重复推理）
  4. 诊断模式 --diagnose-only
"""

import json
import os
import re
import sys
import time
from datasets import load_dataset, concatenate_datasets

# ========================
# 配置
# ========================
SFT_MODEL_PATH = "outputs/sft-qwen2.5-coder-3b-merged"
OUTPUT_DIR = "outputs/dpo-data"
NUM_COMPLETIONS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 1024
TOP_P = 0.95


# ========================
# 从 test_list 推断函数签名
# ========================
def infer_function_name(test_list: list[str]) -> str:
    for test in test_list:
        match = re.search(r'assert\s+(\w+)\s*\(', test)
        if match:
            return match.group(1)
    return ""


# ========================
# 代码提取（核心修复）
# ========================
def extract_code_from_response(response: str) -> str:
    """
    从模型响应中提取代码，按优先级:
      1. <code>...</code> 标签 (SFT 训练格式)
      2. ```python ... ``` markdown 代码块
      3. ``` ... ``` 通用代码块
      4. 去掉 <think> 后剩余内容
      5. 原样返回
    """
    # 1. <code> 标签
    code_match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # 2. ```python
    py_match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if py_match:
        return py_match.group(1).strip()

    # 3. ``` 通用
    generic_match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if generic_match:
        code = generic_match.group(1).strip()
        lines = code.split('\n')
        if lines and lines[0].strip().lower() in ['python', 'py', '']:
            lines = lines[1:]
        return '\n'.join(lines).strip()

    # 4. 去掉 <think> 和 <problem>
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<problem>.*?</problem>', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned and ('def ' in cleaned or 'return ' in cleaned or 'import ' in cleaned):
        return cleaned

    # 5. 原样返回
    return response.strip()


# ========================
# 代码执行
# ========================
def execute_code_with_tests(code: str, test_list: list[str],
                            test_setup_code: str = "") -> dict:
    num_total = len(test_list)
    num_passed = 0
    error_msg = None

    full_code = ""
    if test_setup_code:
        full_code += test_setup_code + "\n"
    full_code += code + "\n"

    for test in test_list:
        exec_code = full_code + "\n" + test
        try:
            exec_globals = {}
            exec(exec_code, exec_globals)
            num_passed += 1
        except Exception as e:
            if error_msg is None:
                error_msg = f"{type(e).__name__}: {e}"
            continue

    return {
        "passed": num_passed == num_total,
        "num_passed": num_passed,
        "num_total": num_total,
        "error": error_msg
    }


# ========================
# Prompt 构造
# ========================
def build_prompts(dataset) -> list[dict]:
    prompts = []
    for item in dataset:
        func_name = infer_function_name(item["test_list"])

        user_content = item["text"]
        if func_name:
            user_content += f"\n\nThe function should be named `{func_name}`."
        if item["test_list"]:
            user_content += f"\n\nExample test case:\n{item['test_list'][0]}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python coding assistant. "
                    "Given a programming problem, think step by step "
                    "and write a correct Python function."
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        prompts.append({
            "task_id": item["task_id"],
            "messages": messages,
            "test_list": item["test_list"],
            "test_setup_code": item.get("test_setup_code", ""),
            "reference_code": item["code"],
        })
    return prompts


# ========================
# vLLM 生成
# ========================
def generate_with_vllm(prompts: list[dict], model_path: str,
                        num_completions: int) -> list[dict]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n=== 加载 vLLM 模型: {model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    formatted_prompts = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(text)

    print(f"  共 {len(formatted_prompts)} 个 prompt，每个生成 {num_completions} 个 completion")

    t0 = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  vLLM 推理完成，耗时 {elapsed:.1f}s")

    results = []
    for prompt_data, output in zip(prompts, outputs):
        completions = [o.text for o in output.outputs]
        results.append({**prompt_data, "completions": completions})
    return results


# ========================
# HF 备用
# ========================
def generate_with_hf(prompts: list[dict], model_path: str,
                      num_completions: int) -> list[dict]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n=== 加载 HF 模型: {model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    results = []
    for i, p in enumerate(prompts):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(prompts)}")
        text = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        completions = []
        for _ in range(num_completions):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            completions.append(tokenizer.decode(new_ids, skip_special_tokens=True))
        results.append({**p, "completions": completions})
    return results


# ========================
# 标注
# ========================
def label_completions(results: list[dict]) -> list[dict]:
    print(f"\n=== 执行测试标注 ===")
    labeled = []
    total_pass = 0
    total_fail = 0

    for i, item in enumerate(results):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(results)}")

        labeled_completions = []
        for comp in item["completions"]:
            code = extract_code_from_response(comp)
            result = execute_code_with_tests(
                code, item["test_list"], item.get("test_setup_code", "")
            )
            labeled_completions.append({
                "response": comp,
                "extracted_code": code,
                "test_result": result,
            })
            if result["passed"]:
                total_pass += 1
            else:
                total_fail += 1

        labeled.append({
            "task_id": item["task_id"],
            "messages": item["messages"],
            "test_list": item["test_list"],
            "reference_code": item["reference_code"],
            "completions": labeled_completions,
        })

    total = total_pass + total_fail
    print(f"\n  标注完成:")
    print(f"  通过: {total_pass}/{total} ({100*total_pass/max(total,1):.1f}%)")
    print(f"  失败: {total_fail}/{total} ({100*total_fail/max(total,1):.1f}%)")
    return labeled


def build_dpo_pairs(labeled_data: list[dict]) -> list[dict]:
    MAX_PAIRS_PER_TASK = 3
    dpo_pairs = []
    skipped_all_pass = 0
    skipped_all_fail = 0

    for item in labeled_data:
        passed = [c for c in item["completions"] if c["test_result"]["passed"]]
        failed = [c for c in item["completions"] if not c["test_result"]["passed"]]

        if not passed:
            skipped_all_fail += 1
            continue
        if not failed:
            skipped_all_pass += 1
            continue

        prompt = item["messages"]
        count = 0
        for chosen_comp in passed:
            for rejected_comp in failed:
                if count >= MAX_PAIRS_PER_TASK:
                    break
                dpo_pairs.append({
                    "task_id": item["task_id"],
                    "prompt": prompt,
                    "chosen": chosen_comp["response"],
                    "rejected": rejected_comp["response"],
                    "chosen_test_result": chosen_comp["test_result"],
                    "rejected_test_result": rejected_comp["test_result"],
                })
                count += 1
            if count >= MAX_PAIRS_PER_TASK:
                break

    print(f"\n=== DPO 数据构造 ===")
    print(f"  总对数: {len(dpo_pairs)}")
    print(f"  跳过 (全部通过): {skipped_all_pass}")
    print(f"  跳过 (全部失败): {skipped_all_fail}")
    print(f"  有效题目: {len(labeled_data) - skipped_all_pass - skipped_all_fail}")
    return dpo_pairs


# ========================
# 诊断
# ========================
def diagnose(results: list[dict], n=5):
    print(f"\n=== 诊断: 前 {n} 个题目 ===")
    for i, item in enumerate(results[:n]):
        print(f"\n{'='*60}")
        print(f"题目 {item['task_id']}: {item['messages'][1]['content'][:80]}...")
        comp = item["completions"][0]
        code = extract_code_from_response(comp)
        print(f"\n[原始响应前 400 字符]")
        print(comp[:400])
        print(f"\n[提取代码前 400 字符]")
        print(code[:400])
        result = execute_code_with_tests(
            code, item["test_list"], item.get("test_setup_code", "")
        )
        print(f"\n[测试] passed={result['passed']}, {result['num_passed']}/{result['num_total']}")
        if result["error"]:
            print(f"[错误] {result['error'][:200]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--model-path", default=SFT_MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-completions", type=int, default=NUM_COMPLETIONS)
    parser.add_argument("--diagnose-only", action="store_true",
                        help="只跑诊断，不做完整标注")
    parser.add_argument("--relabel-only", action="store_true",
                        help="只重新标注已有的 raw_completions（不重新推理）")
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    num_comp = args.num_completions
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print("=== 加载 MBPP 数据集 ===")
    ds = load_dataset("google-research-datasets/mbpp", "full")
    train_data = concatenate_datasets([ds["train"], ds["validation"]])
    print(f"  训练数据: {len(train_data)} 条")

    # 2. 构造 prompts
    prompts = build_prompts(train_data)

    # 3. 获取 completions（生成或加载已有的）
    raw_path = os.path.join(output_dir, "raw_completions.json")

    if args.relabel_only or args.diagnose_only:
        # 只重新标注，不重新推理
        if not os.path.exists(raw_path):
            print(f"  错误: {raw_path} 不存在，无法 relabel")
            return
        print(f"  加载已有 completions: {raw_path}")
        with open(raw_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 但是需要更新 prompts 中的信息到 results
        # (旧的 results 可能没有新的 prompt 格式)
        # results 里的 messages 保持不变，只更新 test 相关字段
        task_id_to_prompt = {p["task_id"]: p for p in prompts}
        for r in results:
            if r["task_id"] in task_id_to_prompt:
                p = task_id_to_prompt[r["task_id"]]
                r["test_list"] = p["test_list"]
                r["test_setup_code"] = p.get("test_setup_code", "")
                r["reference_code"] = p["reference_code"]
    else:
        if os.path.exists(raw_path):
            print(f"\n  发现已有 raw_completions，跳过推理。")
            print(f"  如需重新生成，删除: {raw_path}")
            print(f"  如需只重新标注，加 --relabel-only")
            with open(raw_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 同样更新 test 信息
            task_id_to_prompt = {p["task_id"]: p for p in prompts}
            for r in results:
                if r["task_id"] in task_id_to_prompt:
                    p = task_id_to_prompt[r["task_id"]]
                    r["test_list"] = p["test_list"]
                    r["test_setup_code"] = p.get("test_setup_code", "")
                    r["reference_code"] = p["reference_code"]
        else:
            # 需要生成
            global SFT_MODEL_PATH, NUM_COMPLETIONS
            SFT_MODEL_PATH = model_path
            NUM_COMPLETIONS = num_comp
            if args.backend == "vllm":
                results = generate_with_vllm(prompts, model_path, num_comp)
            else:
                results = generate_with_hf(prompts, model_path, num_comp)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"  保存到: {raw_path}")

    # 4. 诊断模式
    if args.diagnose_only:
        diagnose(results)
        return

    # 5. 标注
    labeled = label_completions(results)
    labeled_path = os.path.join(output_dir, "labeled_completions.json")
    with open(labeled_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False, default=str)

    # 6. DPO 对
    dpo_pairs = build_dpo_pairs(labeled)
    dpo_path = os.path.join(output_dir, "dpo_pairs.json")
    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*50}")
    print(f"完成! DPO 对: {len(dpo_pairs)} 对")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
