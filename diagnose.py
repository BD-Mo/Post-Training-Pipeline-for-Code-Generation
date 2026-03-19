# diagnose.py
import json

with open("outputs/sft_humaneval.jsonl", "r") as f:
    samples = [json.loads(l) for l in f]

# 打印前5条有代码的 completion
count = 0
for s in samples:
    if s["completion"].strip():
        print(f"\n{'='*50}")
        print(f"Task: {s['task_id']}")
        print(f"Completion repr:\n{repr(s['completion'][:300])}")
        count += 1
        if count >= 5:
            break

# 打印结果文件里失败的案例
print(f"\n\n{'='*50}")
print("查看评估结果文件:")
try:
    with open("outputs/sft_humaneval.jsonl_results.jsonl", "r") as f:
        results = [json.loads(l) for l in f]
    
    passed = [r for r in results if r["passed"]]
    failed_with_code = [r for r in results if not r["passed"] and r["completion"].strip()]
    
    print(f"通过: {len(passed)}")
    print(f"失败(有代码): {len(failed_with_code)}")
    
    # 打印前3个失败案例的错误信息
    for r in failed_with_code[:3]:
        print(f"\nTask: {r['task_id']}")
        print(f"Result: {r['result']}")
        print(f"Completion:\n{r['completion'][:200]}")
except Exception as e:
    print(f"读取结果文件失败: {e}")