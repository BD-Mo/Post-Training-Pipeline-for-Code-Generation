"""
把 SFT LoRA adapter 合并到基座模型，保存完整模型
"""
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_PATH  = "outputs/grpo-qwen2.5-coder-3b/final"
SAVE_PATH  = "outputs/grpo-qwen2.5-coder-3b-merged"

print("加载基座模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("加载 LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("合并权重...")
model = model.merge_and_unload()

print(f"保存合并后的模型到 {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(SAVE_PATH)

print("完成！")