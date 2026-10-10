#!/usr/bin/env bash
# One W&B sweep trial per job on one H100 (train 150 steps, eval, log to W&B). Run from the repo
# root; submit a job array to run several trials, e.g. 10 trials with at most 2 at a time:
#   sbatch --account=def-sfischme --array=1-10%2 --time=4:00:00 compute_canada/agent.sh <entity/project/sweep id>
# The sweep id is printed by `wandb sweep sweep_3b_v4.yaml` (run on the login node).
# A trial killed by the time limit shows as crashed in W&B; give 7B more time (--time=10:00:00).
#SBATCH --job-name=sweep-trial
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: sbatch --array=1-10%2 compute_canada/agent.sh <entity/project/sweep id>" >&2
    exit 1
fi
source compute_canada/env.sh
export HF_HUB_OFFLINE=1                                   # weights were downloaded by setup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

wandb agent --count 1 "$1"
