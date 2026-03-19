"""
Stage 2: GRPO 强化学习训练
基于 SFT 模型，用单元测试执行结果作为奖励信号
"""

import re
import os
import torch
import warnings
import tempfile
import subprocess
warnings.filterwarnings("ignore")

from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# ───────────────────────────────
# 配置
# ───────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
SFT_MODEL  = "outputs/sft-qwen2.5-coder-3b-merged"
OUTPUT_DIR = "outputs/grpo-qwen2.5-coder-3b"
MAX_STEPS  = -1    # 冒烟测试，跑通后改为 -1
LOG_STEPS  = 5
SAVE_STEPS = 50


# ───────────────────────────────
# 数据准备
# ───────────────────────────────
def make_prompt(problem: str) -> str:
    return (
        f"<problem>\n{problem}\n</problem>\n"
        f"<think>\nLet me analyze and implement this step by step.\n</think>\n"
        f"<code>\n"
    )


def load_grpo_data():
    from human_eval.data import read_problems
    problems = read_problems()
    records = []
    for task_id, problem in problems.items():
        records.append({
            "task_id":    task_id,
            "prompt":     make_prompt(problem["prompt"]),  # GRPOTrainer 用这列
            "raw_prompt": problem["prompt"],
            "test":       problem["test"],
            "entry_point": problem["entry_point"],
        })
    dataset = Dataset.from_list(records)
    print(f"GRPO 训练数据: {len(dataset)} 条")
    return dataset


# ───────────────────────────────
# 代码提取（和 evaluate.py 保持一致）
# ───────────────────────────────
def extract_code(completion: str, raw_prompt: str) -> str:
    md_blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", completion, re.DOTALL)
    code = md_blocks[0].strip() if md_blocks else completion

    func_name_match = re.search(r"def (\w+)\(", raw_prompt)
    if not func_name_match:
        return ""
    func_name = func_name_match.group(1)
    lines = code.split("\n")

    import_lines = []
    for line in lines:
        if re.match(rf"\s*def {func_name}\(", line):
            break
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ")) \
                and stripped not in raw_prompt:
            import_lines.append(line)

    body_lines = []
    in_function = False
    for line in lines:
        if re.match(rf"\s*def {func_name}\(", line):
            in_function = True
            continue
        if in_function:
            stripped = line.strip()
            if line and not line[0].isspace() and stripped and \
                    not stripped.startswith("#"):
                break
            if stripped.startswith("print(") or \
                    stripped.startswith("assert ") or \
                    stripped.lower().startswith("# test"):
                break
            body_lines.append(line)

    if body_lines:
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        if import_lines:
            indented = ["    " + l.strip() for l in import_lines]
            return "\n".join(indented) + "\n" + "\n".join(body_lines)
        return "\n".join(body_lines)

    first_real = next((l for l in lines if l.strip()), "")
    if first_real.startswith("    ") or first_real.startswith("\t"):
        return code.split("</code>")[0].strip() if "</code>" in code else code.strip()

    return ""


# ───────────────────────────────
# 奖励函数
# ───────────────────────────────
def run_code_safely(code: str, timeout: int = 5) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        success = result.returncode == 0
        error = result.stderr.strip() if not success else ""
        return success, error
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def code_execution_reward(completions, prompts, **kwargs):
    """主奖励：执行单元测试"""
    tests       = kwargs.get("test",        [""] * len(completions))
    entry_pts   = kwargs.get("entry_point", [""] * len(completions))
    raw_prompts = kwargs.get("raw_prompt",  [""] * len(completions))

    rewards = []
    for completion, raw_prompt, test, entry_point in zip(
        completions, raw_prompts, tests, entry_pts
    ):
        body = extract_code(completion, raw_prompt)
        if not body:
            rewards.append(-1.0)
            continue

        full_code = raw_prompt + body + "\n\n" + test + f"\ncheck({entry_point})\n"
        success, error = run_code_safely(full_code, timeout=5)

        if success:
            rewards.append(1.0)
        elif "timeout" in error:
            rewards.append(-0.5)
        else:
            rewards.append(-0.5)

    return rewards


def format_reward(completions, **kwargs):
    """辅助奖励：鼓励输出 </code> 结束标签"""
    return [0.2 if "</code>" in c else 0.0 for c in completions]


# ───────────────────────────────
# 主训练流程
# ───────────────────────────────
def main():
    print("=== GRPO 训练开始 ===")

    dataset = load_grpo_data()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,             # num_generation → num_generations (복수형)
        max_completion_length=512,     # max_new_tokens → max_completion_length
        temperature=0.8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        num_train_epochs=3,
    )

    trainer = GRPOTrainer(
        model=SFT_MODEL,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            code_execution_reward,
            format_reward,
        ],
        peft_config=lora_config,
    )

    print(f"模型加载完成，显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    print("开始训练...")
    trainer.train()

    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"=== GRPO 训练完成，模型保存至 {OUTPUT_DIR}/final ===")


if __name__ == "__main__":
    main()