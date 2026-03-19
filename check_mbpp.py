"""
检查 MBPP 数据集格式，确认可用于 GRPO/DPO 训练
"""
from datasets import load_dataset

ds = load_dataset("google-research-datasets/mbpp", "full")

print("=== 数据集划分 ===")
for split in ds:
    print(f"  {split}: {len(ds[split])} 条")

print("\n=== 字段 ===")
print(ds["train"].column_names)

print("\n=== 前 3 条样本 ===")
for i, item in enumerate(ds["train"].select(range(3))):
    print(f"\n--- 样本 {i} (task_id={item['task_id']}) ---")
    print(f"[text] {item['text']}")
    print(f"[code]\n{item['code']}")
    print(f"[test_list] {item['test_list']}")
    if "test_setup_code" in item:
        print(f"[test_setup_code] {item['test_setup_code']}")

# 统计 test_list 数量分布
test_counts = [len(item["test_list"]) for item in ds["train"]]
print(f"\n=== test_list 统计 ===")
print(f"  总样本数: {len(test_counts)}")
print(f"  平均测试数: {sum(test_counts)/len(test_counts):.1f}")
print(f"  最少: {min(test_counts)}, 最多: {max(test_counts)}")

# 合并 train + validation + test 看总量
total = sum(len(ds[s]) for s in ds)
print(f"\n=== 总计: {total} 条 ===")
print("建议: 用 train+validation 做 RL 训练 (和 HumanEval 完全独立)")
