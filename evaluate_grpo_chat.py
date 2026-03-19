import re
import os
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems

BASE_MODEL  = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_PATH  = "outputs/grpo-qwen2.5-coder-3b-merged"
OUTPUT_FILE = "outputs/grpo_chat_humaneval.jsonl"
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 512


def extract_code(raw, prompt):
    md_blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", raw, re.DOTALL)
    if md_blocks:
        code = md_blocks[0].strip()
    else:
        code = raw

    func_name_match = re.search(r"def (\w+)\(", prompt)
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
                and stripped not in prompt:
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
            if stripped.startswith("print(") or stripped.startswith("assert "):
                break
            body_lines.append(line)

    if body_lines:
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()

        prompt_has_docstring = '"""' in prompt or "'''" in prompt
        if prompt_has_docstring:
            first_nonempty = next(
                (i for i, l in enumerate(body_lines) if l.strip()), None
            )
            if first_nonempty is not None:
                first_line = body_lines[first_nonempty].strip()
                if first_line.startswith('"""') or first_line.startswith("'''"):
                    quote = '"""' if first_line.startswith('"""') else "'''"
                    if first_line.count(quote) >= 2 and len(first_line) > 3:
                        skip_until = first_nonempty + 1
                    else:
                        skip_until = first_nonempty
                        for i in range(first_nonempty + 1, len(body_lines)):
                            if quote in body_lines[i]:
                                skip_until = i + 1
                                break
                    body_lines = body_lines[skip_until:]

        if import_lines:
            indented = ["    " + l.strip() for l in import_lines]
            return "\n".join(indented) + "\n" + "\n".join(body_lines)
        return "\n".join(body_lines)

    first_real = next((l for l in lines if l.strip()), "")
    if first_real.startswith("    ") or first_real.startswith("\t"):
        return code.strip()
    return ""


def generate_solution(model, tokenizer, prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an expert Python programmer. Complete the given Python function. Output only the implementation code."
        },
        {
            "role": "user",
            "content": f"Complete this Python function:\n\n```python\n{prompt}```"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return extract_code(raw, prompt)


def main():
    print("=== GRPO 模型评估（chat template）===")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载完成，显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    problems = read_problems()
    print(f"开始生成，共 {len(problems)} 道题...")

    samples = []
    failed = 0
    for i, (task_id, problem) in enumerate(problems.items()):
        if i % 20 == 0:
            print(f"进度: {i}/{len(problems)}，失败(无代码): {failed}")
        code = generate_solution(model, tokenizer, problem["prompt"])
        if not code:
            failed += 1
        samples.append({"task_id": task_id, "completion": code})

    os.makedirs("outputs", exist_ok=True)
    write_jsonl(OUTPUT_FILE, samples)
    print(f"\n生成完成，失败: {failed}/164")
    print(f"请把 {OUTPUT_FILE} 上传到 Colab 评估")


if __name__ == "__main__":
    main()