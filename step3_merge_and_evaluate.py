"""
Step 3: 合并 LoRA adapter + HumanEval 评估
==========================================
两个功能:
  A. 合并 LoRA adapter 到基础模型 → 输出完整模型
  B. 用 vLLM 跑 HumanEval 推理 → 输出 jsonl 文件
     (Windows 下 human-eval 执行会 timeout，jsonl 需要上传 Colab 评估)

用法:
  # 合并模型
  python step3_merge_and_evaluate.py merge --adapter-path outputs/dpo-qwen2.5-coder-3b/final

  # 生成 HumanEval 结果 (vLLM)
  python step3_merge_and_evaluate.py evaluate --model-path outputs/dpo-qwen2.5-coder-3b-merged --backend vllm

  # 生成 HumanEval 结果 (HF, 兼容 Windows)
  python step3_merge_and_evaluate.py evaluate --model-path outputs/dpo-qwen2.5-coder-3b-merged --backend hf
"""

import argparse
import json
import os
import time
import torch
from typing import Optional


def merge_adapter(adapter_path: str, base_model_path: str, output_path: str):
    """合并 LoRA adapter 到基础模型"""
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"=== 合并 LoRA Adapter ===")
    print(f"  Base model: {base_model_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {output_path}")

    # 加载 base model
    print("  加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 合并在 CPU 上做，节省显存
    )

    # 加载 adapter
    print("  加载 adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # 合并
    print("  合并中...")
    model = model.merge_and_unload()

    # 保存
    print("  保存完整模型...")
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print(f"  完成! 模型保存到: {output_path}")


def evaluate_with_vllm(model_path: str, output_path: str,
                        num_samples: int = 1, temperature: float = 0.0):
    """用 vLLM 跑 HumanEval"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from human_eval.data import read_problems

    print(f"\n=== vLLM HumanEval 评估 ===")
    print(f"  Model: {model_path}")
    print(f"  n={num_samples}, temp={temperature}")

    problems = read_problems()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 构造 prompts
    prompts = []
    task_ids = []
    for task_id, problem in problems.items():
        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Complete the following function. Return ONLY the function body code, no explanation."},
            {"role": "user", "content": problem["prompt"]}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        task_ids.append(task_id)

    # 加载模型
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=512,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    # 推理
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  推理完成，耗时 {elapsed:.1f}s")

    # 保存结果
    samples = []
    for task_id, output in zip(task_ids, outputs):
        for completion in output.outputs:
            # 提取代码
            response = completion.text
            code = extract_function_body(response, problems[task_id]["prompt"])
            samples.append({
                "task_id": task_id,
                "completion": code,
            })

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"  结果保存到: {output_path}")
    print(f"  共 {len(samples)} 个 completion")
    return output_path


def evaluate_with_hf(model_path: str, output_path: str,
                      num_samples: int = 1, temperature: float = 0.0):
    """用 HuggingFace generate 跑 HumanEval（兼容 Windows）"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from human_eval.data import read_problems

    print(f"\n=== HF HumanEval 评估 ===")
    print(f"  Model: {model_path}")

    problems = read_problems()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    samples = []
    for i, (task_id, problem) in enumerate(problems.items()):
        if i % 20 == 0:
            print(f"  进度: {i}/{len(problems)}")

        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Complete the following function. Return ONLY the function body code, no explanation."},
            {"role": "user", "content": problem["prompt"]}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        for _ in range(num_samples):
            with torch.no_grad():
                if temperature > 0:
                    output_ids = model.generate(
                        **inputs, max_new_tokens=512,
                        temperature=temperature, top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                else:
                    output_ids = model.generate(
                        **inputs, max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            code = extract_function_body(response, problem["prompt"])
            samples.append({"task_id": task_id, "completion": code})

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"  结果保存到: {output_path}")
    return output_path


def extract_function_body(response: str, prompt: str) -> str:
    """
    从模型响应中提取可执行的函数补全
    需要处理:
      - markdown 代码块
      - 完整函数定义 (需要去掉函数签名，只保留 body)
      - 纯代码 body
    """
    # 移除 markdown 代码块标记
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            response = parts[1]

    # 如果响应包含完整函数定义，提取 body
    lines = response.strip().split("\n")
    clean_lines = []
    skip_def = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") and ":" in stripped:
            skip_def = True
            continue
        if skip_def and stripped.startswith('"""'):
            # 跳过 docstring
            if stripped.count('"""') >= 2:
                continue  # 单行 docstring
            # 多行 docstring，跳到结束
            for remaining in lines[lines.index(line)+1:]:
                if '"""' in remaining:
                    break
            continue
        clean_lines.append(line)

    result = "\n".join(clean_lines).strip()
    if not result:
        result = response.strip()

    return result + "\n"


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # merge 子命令
    merge_parser = subparsers.add_parser("merge", help="合并 LoRA adapter")
    merge_parser.add_argument("--adapter-path", required=True)
    merge_parser.add_argument("--base-model-path", default=SFT_MODEL_PATH,
                              help="DPO 的 base model 就是 SFT 合并模型")
    merge_parser.add_argument("--output-path", default="outputs/dpo-qwen2.5-coder-3b-merged")

    # evaluate 子命令
    eval_parser = subparsers.add_parser("evaluate", help="HumanEval 评估")
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    eval_parser.add_argument("--output-path", default="outputs/humaneval_results.jsonl")
    eval_parser.add_argument("--num-samples", type=int, default=1)
    eval_parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    if args.command == "merge":
        merge_adapter(args.adapter_path, args.base_model_path, args.output_path)
    elif args.command == "evaluate":
        if args.backend == "vllm":
            evaluate_with_vllm(args.model_path, args.output_path,
                              args.num_samples, args.temperature)
        else:
            evaluate_with_hf(args.model_path, args.output_path,
                            args.num_samples, args.temperature)
    else:
        parser.print_help()


# 默认路径
SFT_MODEL_PATH = "outputs/sft-qwen2.5-coder-3b-merged"

if __name__ == "__main__":
    main()
