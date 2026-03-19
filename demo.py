"""
代码生成 Demo - Gradio + vLLM
==============================
功能:
  1. 输入编程题 → 模型生成代码
  2. 可选输入测试用例 → 自动执行验证
  3. Best-of-N: 生成 N 个答案，自动选出通过测试的

用法:
  pip install gradio
  python demo.py
  # 浏览器打开 http://localhost:7860

面试演示时:
  1. 先启动 demo
  2. 输入 HumanEval 里的题目，展示模型生成代码
  3. 贴入测试用例，展示自动验证
  4. 开启 Best-of-N，展示"生成 5 个选最好的"
"""

import gradio as gr
import subprocess
import tempfile
import sys
import os
import time

# ========================
# 全局模型（启动时加载一次）
# ========================
LLM_ENGINE = None
TOKENIZER = None
MODEL_PATH = "outputs/dpo-merged"  # 改成你要演示的模型


def load_model():
    global LLM_ENGINE, TOKENIZER
    if LLM_ENGINE is not None:
        return

    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"Loading model: {MODEL_PATH}")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
    LLM_ENGINE = LLM(
        model=MODEL_PATH,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    print("Model loaded!")


def generate_code(prompt_text, n=1, temperature=0.0):
    """调用 vLLM 生成代码"""
    from vllm import SamplingParams

    messages = [
        {"role": "system", "content": "You are a Python coding assistant. Complete the following function. Return ONLY the complete function implementation."},
        {"role": "user", "content": prompt_text},
    ]
    text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if temperature == 0:
        params = SamplingParams(n=1, max_tokens=512, temperature=0, stop=["<|endoftext|>", "<|im_end|>"])
    else:
        params = SamplingParams(n=n, max_tokens=512, temperature=temperature, top_p=0.95, stop=["<|endoftext|>", "<|im_end|>"])

    outputs = LLM_ENGINE.generate([text], params)
    return [o.text.strip() for o in outputs[0].outputs]


def execute_code(code, test_code):
    """在沙箱中执行代码 + 测试"""
    full_code = code + "\n\n" + test_code
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp = f.name
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=5
        )
        os.unlink(tmp)
        if result.returncode == 0:
            return True, "All tests passed!"
        else:
            return False, result.stderr[-500:] if result.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp)
        except:
            pass
        return False, "Timeout (>5s)"
    except Exception as e:
        return False, str(e)


# ========================
# Gradio 界面逻辑
# ========================
def single_generate(problem, test_cases):
    """单次生成 (greedy)"""
    load_model()

    t0 = time.time()
    completions = generate_code(problem, n=1, temperature=0)
    gen_time = time.time() - t0
    code = completions[0]

    result_text = f"**Generation time:** {gen_time:.2f}s\n\n"
    result_text += f"```python\n{code}\n```\n\n"

    if test_cases.strip():
        passed, msg = execute_code(code, test_cases)
        if passed:
            result_text += "**Test result: PASSED**"
        else:
            result_text += f"**Test result: FAILED**\n```\n{msg}\n```"
    else:
        result_text += "*No test cases provided — skipping verification*"

    return result_text


def best_of_n_generate(problem, test_cases, n):
    """Best-of-N: 生成 N 个，选通过测试的"""
    load_model()
    n = int(n)

    if not test_cases.strip():
        return "Best-of-N requires test cases to select the best answer. Please provide test cases."

    t0 = time.time()
    completions = generate_code(problem, n=n, temperature=0.8)
    gen_time = time.time() - t0

    result_text = f"**Generated {len(completions)} candidates in {gen_time:.2f}s**\n\n"

    best_code = None
    for i, code in enumerate(completions):
        passed, msg = execute_code(code, test_cases)
        status = "PASSED" if passed else "FAILED"
        result_text += f"---\n**Candidate {i+1}: {status}**\n```python\n{code}\n```\n\n"
        if passed and best_code is None:
            best_code = code

    result_text += "---\n\n"
    if best_code:
        pass_count = sum(1 for c in completions if execute_code(c, test_cases)[0])
        result_text += f"**Result: {pass_count}/{len(completions)} candidates passed tests**\n\n"
        result_text += f"**Selected best answer:**\n```python\n{best_code}\n```"
    else:
        result_text += "**Result: No candidate passed all tests**"

    return result_text


# ========================
# 示例题目
# ========================
EXAMPLES = [
    [
        "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"",
        "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False"
    ],
    [
        "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\n    >>> fibonacci(0)\n    0\n    >>> fibonacci(1)\n    1\n    >>> fibonacci(10)\n    55\n    \"\"\"",
        "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\nassert fibonacci(20) == 6765"
    ],
    [
        "from typing import List\n\ndef remove_duplicates(numbers: List[int]) -> List[int]:\n    \"\"\"Remove duplicates from list while preserving order.\n    >>> remove_duplicates([1, 2, 3, 2, 4, 1])\n    [1, 2, 3, 4]\n    \"\"\"",
        "assert remove_duplicates([1, 2, 3, 2, 4, 1]) == [1, 2, 3, 4]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1, 1, 1]) == [1]"
    ],
]


# ========================
# 构建界面
# ========================
def build_demo():
    with gr.Blocks(title="Code Generation Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Qwen2.5-Coder-3B Code Generation Demo\n"
            "**Pipeline:** Base → SFT (+30pp) → DPO (+3pp) | "
            "HumanEval Pass@1: 20.7% → 51.2% | Pass@10: 58.5% → 92.7%"
        )

        with gr.Row():
            with gr.Column(scale=1):
                problem_input = gr.Textbox(
                    label="Programming Problem",
                    placeholder="Paste a function signature with docstring...",
                    lines=10,
                )
                test_input = gr.Textbox(
                    label="Test Cases (optional, needed for verification & Best-of-N)",
                    placeholder="assert my_func(1, 2) == 3\nassert my_func(0, 0) == 0",
                    lines=5,
                )
                with gr.Row():
                    greedy_btn = gr.Button("Generate (Greedy)", variant="primary")
                    bon_btn = gr.Button("Best-of-N", variant="secondary")
                    n_slider = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="N")

            with gr.Column(scale=1):
                output = gr.Markdown(label="Result")

        gr.Examples(
            examples=EXAMPLES,
            inputs=[problem_input, test_input],
            label="Example Problems",
        )

        greedy_btn.click(fn=single_generate, inputs=[problem_input, test_input], outputs=output)
        bon_btn.click(fn=best_of_n_generate, inputs=[problem_input, test_input, n_slider], outputs=output)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
