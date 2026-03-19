"""
数据准备脚本
数据集：CodeFeedback-Filtered-Instruction
格式：把原始数据转换成 <problem>/<think>/<code> 三段式格式
"""

from datasets import load_dataset
import json
import os

def format_example(example):
    """
    把原始数据转换成我们的训练格式
    输入：{"query": "...", "answer": "..."}
    输出：带 special token 的训练文本
    """
    problem = example["query"].strip()
    raw_answer = example["answer"].strip()

    # 尝试从 answer 里分离思考过程和代码
    # CodeFeedback 的 answer 通常是直接的代码+解释混合
    # 我们做一个简单的分离：把解释当作 think，把代码块提取出来
    
    import re
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", raw_answer, re.DOTALL)
    
    if code_blocks:
        code = code_blocks[0].strip()
        # 去掉代码块后剩下的文字作为思考过程
        think = re.sub(r"```(?:python)?\n.*?```", "", raw_answer, flags=re.DOTALL).strip()
        if not think:
            think = "Let me analyze the problem and implement a solution step by step."
    else:
        # 没有代码块，整体作为代码
        code = raw_answer
        think = "Let me analyze the problem and implement a solution step by step."

    # 控制长度，避免超出 max_seq_length
    if len(think) > 500:
        think = think[:500] + "..."
    if len(code) > 1000:
        return None  # 过长的样本直接跳过

    text = (
        f"<problem>\n{problem}\n</problem>\n"
        f"<think>\n{think}\n</think>\n"
        f"<code>\n{code}\n</code>"
    )
    return {"text": text}


def prepare_dataset(num_samples=5000, output_path="data/train.jsonl"):
    """
    下载并处理数据集
    num_samples: 取前 N 条，5000 条对 SFT 冷启动已经够用
    """
    print("正在下载数据集 CodeFeedback-Filtered-Instruction ...")
    dataset = load_dataset(
        "m-a-p/CodeFeedback-Filtered-Instruction",
        split="train"
    )
    
    print(f"原始数据集大小: {len(dataset)}")
    
    # 过滤：只保留 Python 相关的样本
    dataset = dataset.filter(
        lambda x: "python" in x["answer"].lower() or 
                  "Python" in x["query"] or
                  "```python" in x["answer"]
    )
    print(f"过滤后 Python 样本数: {len(dataset)}")
    
    # 取前 num_samples 条
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # 格式转换
    processed = []
    skipped = 0
    for example in dataset:
        result = format_example(example)
        if result is not None:
            processed.append(result)
        else:
            skipped += 1
    
    print(f"处理完成: {len(processed)} 条，跳过 {skipped} 条")
    
    # 保存
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"已保存到 {output_path}")
    
    # 打印一个样本预览
    print("\n=== 样本预览 ===")
    print(processed[0]["text"][:500])
    print("...")
    
    return processed


if __name__ == "__main__":
    prepare_dataset(num_samples=5000)