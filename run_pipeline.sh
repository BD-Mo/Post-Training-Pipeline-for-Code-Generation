#!/bin/bash
# ==============================================
# Qwen2.5-Coder-3B Post-Training Pipeline
# 一键执行全流程
# ==============================================
set -e

echo "=========================================="
echo "Step 0: 检查 MBPP 数据集"
echo "=========================================="
python check_mbpp.py

echo ""
echo "=========================================="
echo "Step 1: 生成 DPO 偏好数据"
echo "=========================================="
# 检测 vLLM 是否可用
if python -c "import vllm" 2>/dev/null; then
    echo "  检测到 vLLM，使用 vLLM 后端"
    python step1_generate_dpo_data.py --backend vllm
else
    echo "  未检测到 vLLM，使用 HF 后端 (较慢)"
    python step1_generate_dpo_data.py --backend hf
fi

echo ""
echo "=========================================="
echo "Step 2: DPO 训练"
echo "=========================================="
python step2_dpo_train.py

echo ""
echo "=========================================="
echo "Step 3a: 合并 LoRA adapter"
echo "=========================================="
python step3_merge_and_evaluate.py merge \
    --adapter-path outputs/dpo-qwen2.5-coder-3b/final

echo ""
echo "=========================================="
echo "Step 3b: HumanEval 推理"
echo "=========================================="
if python -c "import vllm" 2>/dev/null; then
    python step3_merge_and_evaluate.py evaluate \
        --model-path outputs/dpo-qwen2.5-coder-3b-merged \
        --backend vllm \
        --output-path outputs/humaneval_dpo_results.jsonl
else
    python step3_merge_and_evaluate.py evaluate \
        --model-path outputs/dpo-qwen2.5-coder-3b-merged \
        --backend hf \
        --output-path outputs/humaneval_dpo_results.jsonl
fi

echo ""
echo "=========================================="
echo "流程完成!"
echo "=========================================="
echo ""
echo "生成的 HumanEval 结果: outputs/humaneval_dpo_results.jsonl"
echo ""
echo "下一步: 上传 jsonl 到 Colab 评估 Pass@1"
echo "  python colab_evaluate.py --input humaneval_dpo_results.jsonl"
