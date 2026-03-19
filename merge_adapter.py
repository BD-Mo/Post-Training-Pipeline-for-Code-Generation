"""
合并 LoRA adapter 到基础模型
用法:
  python merge_adapter.py --adapter outputs/sft/final --output outputs/sft-merged
  python merge_adapter.py --adapter outputs/dpo/final --base outputs/sft-merged --output outputs/dpo-merged
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Base model")
    parser.add_argument("--output", required=True, help="Output merged model path")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Base:    {args.base}")
    print(f"Adapter: {args.adapter}")
    print(f"Output:  {args.output}")

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")

    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging...")
    model = model.merge_and_unload()

    print("Saving...")
    model.save_pretrained(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.output)

    print(f"Done! Merged model at: {args.output}")


if __name__ == "__main__":
    main()
