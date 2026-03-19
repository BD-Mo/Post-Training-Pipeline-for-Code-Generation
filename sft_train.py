"""
SFT 训练 - 用标准 chat template 格式
======================================
关键设计: 训练格式 = 评估格式 = 标准 chat template
不再用自定义 <problem>/<think>/<code> 格式

数据: CodeFeedback-Filtered-Instruction, 筛选 Python 样本
格式: system + user(问题) + assistant(代码)

用法:
  python sft_train.py
  python sft_train.py --max-samples 2000 --epochs 2
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ========================
# 配置
# ========================
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
OUTPUT_DIR = "outputs/sft"
MAX_SAMPLES = 3000
NUM_EPOCHS = 2
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1536

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = "You are a Python coding assistant. Complete the following function. Return ONLY the complete function implementation."


def prepare_data(tokenizer, max_samples):
    """
    从 CodeFeedback-Filtered-Instruction 准备 SFT 数据
    筛选 Python 代码，格式化为 chat template
    """
    print("加载 CodeFeedback 数据集...")
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    print(f"  原始数据: {len(ds)} 条")

    # 筛选 Python 样本
    def is_python(example):
        answer = example.get("answer", "")
        query = example.get("query", "")
        # 包含 python 关键字或 def/import 等 Python 特征
        text = (answer + query).lower()
        if "python" in text:
            return True
        if "```python" in answer:
            return True
        if "def " in answer and "return " in answer:
            return True
        return False

    ds = ds.filter(is_python)
    print(f"  Python 样本: {len(ds)} 条")

    # 限制数量
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    print(f"  使用: {len(ds)} 条")

    # 格式化为 chat messages
    formatted = []
    for item in ds:
        query = item["query"].strip()
        answer = item["answer"].strip()

        # 跳过太短的答案
        if len(answer) < 50:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]

        # 用 chat template 编码，检查长度
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(text, return_length=True)["length"][0]
        if tokens > MAX_SEQ_LENGTH:
            continue

        formatted.append({"messages": messages})

    print(f"  格式化后: {len(formatted)} 条 (过滤掉过长的)")
    return Dataset.from_list(formatted)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Training")
    print("=" * 60)
    print(f"  Base model: {args.base_model}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Data
    dataset = prepare_data(tokenizer, args.max_samples)

    # 3. Model
    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # 4. LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    # 5. Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=args.lr,
        max_length=MAX_SEQ_LENGTH,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 7. Train
    print("\n开始训练...")
    result = trainer.train()

    # 8. Save
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print(f"SFT 完成!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Adapter: {final_path}")
    print(f"  下一步: python merge_adapter.py --adapter {final_path} --output outputs/sft-merged")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
