#!/usr/bin/env bash
# Build the inputs a W&B sweep needs:
#   <out_dir>/histories.jsonl            shared training prompts (collect_histories.py, 64 lists x 150 steps),
#                                        collected by <collector> (default: the base model itself)
#   <out_dir>/base_eval/eval_steps.csv   the base model's reference curve (evaluate.py, same settings as the trials)
# Each step is skipped if its output already exists, so it is safe to rerun.
# batch_size (default 16) is how many lists are generated at once; use the sweep YAML's batch_size.
# v3 collects with a trained v2 model, merged first with merge_adapter.py.
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./prepare_sweep.sh Qwen/Qwen2.5-3B-Instruct sweeps/3b-v3 16 sweeps/3b-v3/collector
#   ./prepare_sweep.sh Qwen/Qwen2.5-7B-Instruct sweeps/7b-v3 8 sweeps/7b-v3/collector
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    echo "usage: $0 <model> <out_dir> [batch_size] [collector]" >&2
    exit 1
fi
model=$1
out=$2
batch=${3:-16}
collector=${4:-$model}
cd "$(dirname "$0")"
mkdir -p "$out"

if [ ! -f "$out/histories.jsonl" ]; then
    echo "Building shared histories with $collector"
    python -u collect_histories.py --model "$collector" --steps 150 --episodes 64 --batch-size "$batch" \
        --out "$out/histories.partial.jsonl"
    mv "$out/histories.partial.jsonl" "$out/histories.jsonl"
fi

# Keep these in step with eval_episodes / eval_length / batch_size / seed in sweep_*.yaml.
if [ ! -f "$out/base_eval/eval_steps.csv" ]; then
    echo "Evaluating the base model"
    python -u evaluate.py --models "$model" --labels base --episodes 16 --steps 150 --batch-size "$batch" --seed 0 \
        --out-dir "$out/base_eval"
fi
echo "Ready: $out/histories.jsonl and $out/base_eval/eval_steps.csv"
