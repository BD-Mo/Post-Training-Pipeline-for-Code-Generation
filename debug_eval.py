import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from human_eval.data import read_problems

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
SFT_MODEL  = "outputs/grpo-qwen2.5-coder-3b-merged"

# ───────────────────────────────
# 提取函数
# ───────────────────────────────
def extract_code(raw, prompt):
    # Step 1: markdown 代码块优先
    md_blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", raw, re.DOTALL)
    code = md_blocks[0].strip() if md_blocks else raw

    # Step 2: 从 prompt 找函数名
    func_name_match = re.search(r"def (\w+)\(", prompt)
    if not func_name_match:
        return ""
    func_name = func_name_match.group(1)

    lines = code.split("\n")

    # Step 3: 收集函数 def 之前的 import 语句
# 只收集 prompt 里没有的 import
    import_lines = []
    for line in lines:
        if re.match(rf"\s*def {func_name}\(", line):
            break
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ")) and stripped not in prompt:
            import_lines.append(line)

    # Step 4: 提取函数体
    body_lines = []
    in_function = False
    for line in lines:
        if re.match(rf"\s*def {func_name}\(", line):
            in_function = True
            continue
        if in_function:
            # 停止条件：顶层非缩进代码（排除空行和注释）
            if line and not line[0].isspace() and line.strip() and \
               not line.strip().startswith("#"):
                break
            # 停止条件：测试代码（print / assert / # Test）
            stripped = line.strip()
            if stripped.startswith("print(") or \
               stripped.startswith("assert ") or \
               stripped.lower().startswith("# test"):
                break
            body_lines.append(line)

    if body_lines:
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()

        # 如果 prompt 已经有 docstring，去掉 body_lines 开头重复的 docstring
        # 检测：body_lines 第一个非空行是否是 """ 开头
        prompt_has_docstring = '"""' in prompt or "'''" in prompt

        if prompt_has_docstring:
            # 找到第一个非空行
            first_nonempty = next(
                (i for i, l in enumerate(body_lines) if l.strip()), None
            )
            if first_nonempty is not None:
                first_line = body_lines[first_nonempty].strip()
                # 如果开头是 docstring，跳过整个 docstring 块
                if first_line.startswith('"""') or first_line.startswith("'''"):
                    quote = '"""' if first_line.startswith('"""') else "'''"
                    # 找到 docstring 结束位置
                    in_docstring = True
                    skip_until = first_nonempty
                    # 单行 docstring（开头和结尾都有引号）
                    if first_line.count(quote) >= 2 and len(first_line) > 3:
                        skip_until = first_nonempty + 1
                    else:
                        for i in range(first_nonempty + 1, len(body_lines)):
                            if quote in body_lines[i]:
                                skip_until = i + 1
                                break
                    body_lines = body_lines[skip_until:]

        if import_lines:
            indented = ["    " + l.strip() for l in import_lines]
            return "\n".join(indented) + "\n" + "\n".join(body_lines)
        return "\n".join(body_lines)

    # Step 5: 没找到函数定义，检查是否已经是纯函数体
    first_real_line = next((l for l in lines if l.strip()), "")
    if first_real_line.startswith("    ") or first_real_line.startswith("\t"):
        return code.split("</code>")[0].strip() if "</code>" in code else code.strip()

    return ""


# ───────────────────────────────
# 加载模型
# ───────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL, dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, trust_remote_code=True)
model.eval()
print(f"模型加载完成，显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

problems = read_problems()

# ───────────────────────────────
# 生成 + 提取，看前3题
# ───────────────────────────────
for task_id, problem in list(problems.items())[:5]:
    prompt = problem["prompt"]

    input_text = (
        f"<problem>\n{prompt}\n</problem>\n"
        f"<think>\nLet me analyze and implement this step by step.\n</think>\n"
        f"<code>\n"
    )

    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    extracted = extract_code(raw, prompt)

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"--- 完整原始输出 ---")
    print(raw)
    print(f"--- 提取结果 ---")
    if extracted:
        print(extracted)
    else:
        print("⚠️  空字符串（无法提取代码）")
    print(f"--- HumanEval completion 拼接预览 ---")
    print(prompt + extracted)