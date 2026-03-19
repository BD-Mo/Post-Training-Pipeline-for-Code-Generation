"""
DPO 训练
========
从 SFT 模型出发，用偏好数据做 DPO

用法:
  python dpo_train.py --model outputs/sft-merged --data outputs/dpo-data/dpo_pairs.json
"""

import json
import os
import torch
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

BETA = 0.1
LEARNING_RATE = 5e-6
NUM_EPOCHS = 2
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 1536

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SFT merged model path")
    parser.add_argument("--data", required=True, help="DPO pairs json path")
    parser.add_argument("--output", default="outputs/dpo")
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    print("=" * 60)
    print("DPO Training")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Data:  {args.data}")
    print(f"  Beta:  {args.beta}")
    print(f"  LR:    {args.lr}")

    # Load data
    with open(args.data) as f:
        raw = json.load(f)
    print(f"  Pairs: {len(raw)}")

    dataset = Dataset.from_list([
        {"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"]}
        for item in raw
    ])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none",
    )

    # DPO config
    training_args = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_length=MAX_LENGTH,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\nTraining...")
    result = trainer.train()

    final_path = os.path.join(args.output, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print(f"DPO Done!")
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Adapter: {final_path}")
    print(f"  Next: python merge_adapter.py --adapter {final_path} --base {args.model} --output outputs/dpo-merged")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
