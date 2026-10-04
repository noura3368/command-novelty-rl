#!/usr/bin/env bash
# Build the inputs a W&B sweep needs, with the base model:
#   <out_dir>/histories.jsonl            shared training prompts (collect_histories.py, 100 steps)
#   <out_dir>/base_eval/eval_steps.csv   reference curve (evaluate.py, same settings as the sweep's trials)
# Each step is skipped if its output already exists, so it is safe to rerun.
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./prepare_sweep.sh Qwen/Qwen2.5-3B-Instruct sweeps/3b
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <model> <out_dir>" >&2
    exit 1
fi
model=$1
out=$2
cd "$(dirname "$0")"
mkdir -p "$out"

if [ ! -f "$out/histories.jsonl" ]; then
    echo "Building shared histories with $model"
    python -u collect_histories.py --model "$model" --steps 100 --episodes 64 --batch-size 16 \
        --out "$out/histories.partial.jsonl"
    mv "$out/histories.partial.jsonl" "$out/histories.jsonl"
fi

# Keep these in step with eval_episodes / eval_length / batch_size / seed in sweep_*.yaml.
if [ ! -f "$out/base_eval/eval_steps.csv" ]; then
    echo "Evaluating the base model"
    python -u evaluate.py --models "$model" --labels base --episodes 16 --steps 60 --batch-size 16 --seed 0 \
        --out-dir "$out/base_eval"
fi
echo "Ready: $out/histories.jsonl and $out/base_eval/eval_steps.csv"
