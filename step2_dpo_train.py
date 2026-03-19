"""
Step 2: DPO 训练
================
从 SFT 合并模型出发，用 Step 1 生成的偏好数据训练 DPO

技术要点 (面试可说):
  - DPO 直接优化偏好，不需要单独的 reward model
  - 偏好数据来自 SFT 模型自身的 on-policy 生成 + 自动化测试标注
  - 用 LoRA 微调降低显存占用，3B 模型在单卡 32GB 上可训
  - beta 参数控制偏离参考模型的程度: 太小容易过拟合，太大学不动
"""

import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

# ========================
# 配置
# ========================
SFT_MODEL_PATH = "outputs/sft-qwen2.5-coder-3b-merged"
DPO_DATA_PATH = "outputs/dpo-data/dpo_pairs.json"
OUTPUT_DIR = "outputs/dpo-qwen2.5-coder-3b"

# DPO 超参
BETA = 0.1                  # KL 惩罚系数，控制偏离 SFT 模型的程度
LEARNING_RATE = 5e-6         # DPO 一般用比 SFT 更小的 lr
NUM_EPOCHS = 2
BATCH_SIZE = 2               # 受显存限制
GRADIENT_ACCUMULATION = 4    # 等效 batch = 2 * 4 = 8
MAX_LENGTH = 1536            # prompt + response 最大长度
MAX_PROMPT_LENGTH = 512

# LoRA 配置 - 和 SFT 保持一致的 rank
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_dpo_data(path: str) -> Dataset:
    """
    加载 DPO 偏好对数据，转换为 DPOTrainer 需要的格式
    DPOTrainer 需要: prompt (str), chosen (str), rejected (str)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"加载了 {len(raw_data)} 个 DPO 对")

    # 转换格式
    processed = []
    for item in raw_data:
        # prompt 是 messages 列表，DPOTrainer 需要它来构造 chat template
        # 我们直接传 messages，让 tokenizer 的 chat template 处理
        processed.append({
            "prompt": item["prompt"],      # list of dicts (messages)
            "chosen": item["chosen"],      # chosen response text
            "rejected": item["rejected"],  # rejected response text
        })

    dataset = Dataset.from_list(processed)
    print(f"数据集大小: {len(dataset)}")
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=SFT_MODEL_PATH)
    parser.add_argument("--data-path", default=DPO_DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("=" * 60)
    print("DPO 训练")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  数据: {args.data_path}")
    print(f"  输出: {args.output_dir}")
    print(f"  beta: {args.beta}")
    print(f"  lr: {args.lr}")
    print(f"  epochs: {args.epochs}")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型
    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",  # 如果支持
    )

    # 3. 添加 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    # 4. 加载数据
    dataset = load_dpo_data(args.data_path)

    # 5. DPO 训练配置
    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",  # 如果有 wandb 可以改成 "wandb"
        # 重要: 让 DPOTrainer 处理 chat template 格式
        # dataset_num_proc=4,  # 数据预处理并行
    )

    # 6. 创建 DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 7. 训练
    print("\n开始训练...")
    train_result = trainer.train()

    # 8. 保存
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  LoRA adapter 保存到: {final_path}")
    print(f"  下一步: 运行 step3_merge_and_evaluate.py 合并并评估")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
