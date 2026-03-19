"""
GRPO v2: 用 MBPP 数据集做 GRPO 训练 (备选方案)
================================================
如果 DPO 效果不理想，可以尝试 GRPO

和 v1 (失败版本) 的区别:
  - 数据集从 HumanEval 164 条 → MBPP 374+ 条，避免训练集 = 评估集
  - 奖励函数基于代码执行结果，不是简单的格式检查
  - 降低了 max_completion_length 避免 OOM
  - 添加了格式保持奖励，防止格式退化

注意: trl 0.29.0 下 GRPOConfig 用 num_generations (复数) 和 max_completion_length
"""

import json
import os
import sys
import signal
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

# ========================
# 配置
# ========================
SFT_MODEL_PATH = "outputs/sft-qwen2.5-coder-3b-merged"
OUTPUT_DIR = "outputs/grpo-qwen2.5-coder-3b-v2"
EXEC_TIMEOUT = 10

# GRPO 超参
NUM_GENERATIONS = 4      # 每个 prompt 生成几个 completion (显存瓶颈)
MAX_COMPLETION_LENGTH = 512
LEARNING_RATE = 1e-6     # GRPO 用很小的 lr
NUM_EPOCHS = 1
BATCH_SIZE = 1           # GRPO 显存消耗很大
GRADIENT_ACCUMULATION = 8

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ========================
# 奖励函数
# ========================
def execute_code_safe(code: str, test_list: list[str],
                      test_setup_code: str = "") -> dict:
    """安全执行代码并返回测试结果"""
    num_total = len(test_list)
    num_passed = 0

    try:
        full_code = ""
        if test_setup_code:
            full_code += test_setup_code + "\n"
        full_code += code + "\n"

        for test in test_list:
            try:
                exec(full_code + test, {"__builtins__": __builtins__}, {})
                num_passed += 1
            except:
                continue
    except:
        pass

    return {"passed": num_passed == num_total, "ratio": num_passed / max(num_total, 1)}


def extract_code(response: str) -> str:
    """从模型输出提取代码"""
    if "```python" in response:
        blocks = response.split("```python")
        if len(blocks) > 1:
            return blocks[1].split("```")[0].strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 2:
            return parts[1].strip()
    return response.strip()


def make_reward_fn(dataset_items: list[dict]):
    """
    创建奖励函数闭包
    GRPO 的 reward function 签名: (completions, prompts) -> list[float]
    但 trl 0.29 的签名是 (completions, **kwargs) -> list[float]
    需要根据你的 trl 版本调整
    """
    # 建立 prompt -> test_list 的映射
    prompt_to_tests = {}
    for item in dataset_items:
        # 用题目文本作为 key
        prompt_to_tests[item["text"]] = {
            "test_list": item["test_list"],
            "test_setup_code": item.get("test_setup_code", ""),
        }

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """
        奖励函数:
          - 代码执行通过所有测试: +1.0
          - 部分通过: 通过比例 * 0.5
          - 全部失败: -0.5
          - 有代码块格式: +0.1 (格式保持奖励)
        """
        prompts = kwargs.get("prompts", kwargs.get("prompt", []))
        rewards = []

        for i, completion in enumerate(completions):
            reward = 0.0

            # 格式奖励: 鼓励使用代码块
            if "```python" in completion or "```" in completion:
                reward += 0.1

            # 提取代码
            code = extract_code(completion)

            # 执行奖励
            # 尝试匹配对应的测试用例
            if i < len(prompts):
                prompt_text = prompts[i]
                # 在映射中查找
                matched = False
                for key, tests in prompt_to_tests.items():
                    if key in prompt_text:
                        result = execute_code_safe(
                            code, tests["test_list"], tests["test_setup_code"]
                        )
                        if result["passed"]:
                            reward += 1.0
                        elif result["ratio"] > 0:
                            reward += result["ratio"] * 0.5
                        else:
                            reward -= 0.5
                        matched = True
                        break

                if not matched:
                    # 无法匹配测试用例，给中性奖励
                    reward += 0.0

            rewards.append(reward)

        return rewards

    return reward_fn


# ========================
# 数据准备
# ========================
def prepare_mbpp_prompts(tokenizer) -> Dataset:
    """将 MBPP 题目转换为 GRPO 需要的 prompt 格式"""
    ds = load_dataset("google-research-datasets/mbpp", "full")
    data = concatenate_datasets([ds["train"], ds["validation"]])

    formatted = []
    for item in data:
        messages = [
            {
                "role": "system",
                "content": "You are a Python coding assistant. Think step by step and write a correct Python function."
            },
            {
                "role": "user",
                "content": item["text"]
            }
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append({
            "prompt": prompt_text,
            "original_text": item["text"],
        })

    return Dataset.from_list(formatted), list(data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=SFT_MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO v2 训练 (MBPP)")
    print("=" * 60)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 数据
    dataset, raw_items = prepare_mbpp_prompts(tokenizer)
    print(f"训练数据: {len(dataset)} 条")

    # 3. 奖励函数
    reward_fn = make_reward_fn(raw_items)

    # 4. LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    # 5. GRPO 配置
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # 6. 模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 7. Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 8. 训练
    print("\n开始训练...")
    train_result = trainer.train()

    # 9. 保存
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\nGRPO 训练完成!")
    print(f"  Loss: {train_result.training_loss:.4f}")
    print(f"  Adapter 保存到: {final_path}")


if __name__ == "__main__":
    main()
