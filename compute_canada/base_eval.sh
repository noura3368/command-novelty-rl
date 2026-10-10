#!/usr/bin/env bash
# Base-model eval for a sweep (prepare_sweep.sh) on one H100. Run from the repo root, after
# copying the training prompts to <out_dir>/histories.jsonl (otherwise prepare_sweep.sh would collect new ones):
#   sbatch --account=def-sfischme compute_canada/base_eval.sh Qwen/Qwen2.5-3B-Instruct sweeps/3b-v4
#   sbatch --account=def-sfischme compute_canada/base_eval.sh Qwen/Qwen2.5-7B-Instruct sweeps/7b-v4
# Writes <out_dir>/base_eval/eval_steps.csv, which the sweep YAML's base_eval points to.
#SBATCH --job-name=base-eval
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: sbatch compute_canada/base_eval.sh <model> <out_dir>" >&2
    exit 1
fi
source compute_canada/env.sh
export HF_HUB_OFFLINE=1                                   # weights were downloaded by setup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# eval batch 16 = batch_size in sweep_*_v4.yaml
./prepare_sweep.sh "$1" "$2" 16
