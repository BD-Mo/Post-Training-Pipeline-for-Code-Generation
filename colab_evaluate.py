"""
Colab 评估脚本
=============
Windows 下 human-eval 的代码执行会 timeout
需要在 Colab (Linux) 上运行这个脚本来计算 Pass@1

用法:
  1. 上传 humaneval_results.jsonl 到 Colab
  2. pip install human-eval
  3. python colab_evaluate.py --input humaneval_results.jsonl

或者在 Colab cell 里直接跑:
  !pip install human-eval
  from human_eval.evaluation import evaluate_functional_correctness
  results = evaluate_functional_correctness("humaneval_results.jsonl")
  print(results)
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="jsonl 文件路径")
    parser.add_argument("--k", type=int, nargs="+", default=[1],
                        help="计算 Pass@k，默认 k=1")
    args = parser.parse_args()

    from human_eval.evaluation import evaluate_functional_correctness
    results = evaluate_functional_correctness(
        args.input,
        k=args.k,
        n_workers=4,
        timeout=10.0,
    )

    print("\n" + "=" * 40)
    print("HumanEval 评估结果")
    print("=" * 40)
    for key, val in results.items():
        print(f"  {key}: {val*100:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
