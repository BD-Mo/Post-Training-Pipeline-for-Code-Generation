# Qwen2.5-Coder-3B Post-Training Pipeline

Full post-training pipeline (SFT → DPO) for code generation, with Pass@k gap analysis and vLLM-accelerated inference.

## Results

| Stage | Pass@1 (greedy) | Pass@1 (sample) | Pass@5 | Pass@10 |
|-------|:-:|:-:|:-:|:-:|
| Base | 20.7% | 24.2% | 50.8% | 58.5% |
| + SFT | **50.6%** | 51.0% | 85.7% | 92.1% |
| + DPO | 51.2% | **54.1%** | **87.3%** | **92.7%** |

All numbers evaluated on HumanEval (164 problems) using the same script and prompt format.

## Approach

**SFT** — LoRA fine-tuning on 2982 Python samples from CodeFeedback-Filtered-Instruction. Standard chat template format so training and evaluation are consistent. Achieves +29.9pp on greedy Pass@1.

**Pass@k gap analysis** — After SFT, Pass@1=51% but Pass@10=92%, a 41pp gap. This means the model can solve 92% of problems but only picks the right answer ~half the time. This gap justifies preference optimization.

**DPO** — Used vLLM to generate 3712 completions (464 MBPP problems × 8 each) from the SFT model in 87 seconds. Each completion is executed against test cases in a sandboxed subprocess. Pass → chosen, fail → rejected. 657 preference pairs constructed. DPO narrows the sampling gap and pushes Pass@1 (sample) to 54.1%.

**Demo** — Gradio interface with greedy generation and Best-of-N selection, showcasing the Pass@1 vs Pass@k gap in real time.

## Project Structure

```
evaluate.py           # Unified evaluation (Pass@1 and Pass@k)
sft_train.py          # SFT with LoRA on CodeFeedback
merge_adapter.py      # Merge LoRA adapter into base model
generate_dpo_data.py  # vLLM batch inference + execution-based labeling
dpo_train.py          # DPO training
demo.py               # Gradio demo with Best-of-N
```

## Quick Start

### Environment

```bash
# WSL2 Ubuntu 24.04, RTX 5090 32GB
conda create -n llm-finetune python=3.11 -y
conda activate llm-finetune
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install trl==0.29.0 peft transformers datasets accelerate human-eval gradio
```

### Run Pipeline

```bash
# Phase 0: Evaluate base model
python evaluate.py --model Qwen/Qwen2.5-Coder-3B-Instruct --output results/base_p1.jsonl
python evaluate.py --model Qwen/Qwen2.5-Coder-3B-Instruct --n 10 --temp 0.8 --output results/base_p10.jsonl
python evaluate.py --eval-only --input results/base_p10.jsonl --k 1 5 10

# Phase 1: SFT
python sft_train.py
python merge_adapter.py --adapter outputs/sft/final --output outputs/sft-merged
python evaluate.py --model outputs/sft-merged --output results/sft_p1.jsonl

# Phase 2: Measure Pass@k gap (decides whether to do RL)
python evaluate.py --model outputs/sft-merged --n 10 --temp 0.8 --output results/sft_p10.jsonl
python evaluate.py --eval-only --input results/sft_p10.jsonl --k 1 5 10

# Phase 3: DPO
python generate_dpo_data.py --model outputs/sft-merged
python dpo_train.py --model outputs/sft-merged --data outputs/dpo-data/dpo_pairs.json
python merge_adapter.py --adapter outputs/dpo/final --base outputs/sft-merged --output outputs/dpo-merged
python evaluate.py --model outputs/dpo-merged --output results/dpo_p1.jsonl
python evaluate.py --model outputs/dpo-merged --n 10 --temp 0.8 --output results/dpo_p10.jsonl
python evaluate.py --eval-only --input results/dpo_p10.jsonl --k 1 5 10

# Demo
python demo.py
```

## Technical Details

| Component | Detail |
|-----------|--------|
| Base model | Qwen2.5-Coder-3B-Instruct |
| SFT data | CodeFeedback-Filtered-Instruction, 2982 Python samples |
| SFT method | LoRA r=16, α=32, 7 target modules (QKV + O + MLP) |
| DPO data | MBPP 464 problems, 3712 completions, 657 preference pairs |
| DPO labeling | Automated execution-based: subprocess sandbox, 5s timeout |
| DPO config | β=0.05, lr=1e-5, 2 epochs |
| Inference | vLLM v0.17.1, continuous batching, 3712 completions in 87s |
| Evaluation | HumanEval 164 problems, unified script for all stages |
| Hardware | RTX 5090 32GB via WSL2 |

## Key Insight

The Pass@k gap analysis is the core analytical contribution. After SFT:

- **Pass@1 = 51%** — the model picks the correct answer about half the time
- **Pass@10 = 92%** — but it *can* produce a correct answer for 92% of problems

This 41pp gap means the model has the knowledge but lacks the consistency. DPO addresses this by learning from its own successes and failures, narrowing the gap and improving sampling Pass@1 by +3.1pp.

## License

MIT
