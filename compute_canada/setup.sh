#!/usr/bin/env bash
# One-time setup on a cluster LOGIN node (compute nodes may have no internet):
#   1. virtualenv at $VENV with requirements.txt
#   2. model weights downloaded into $HF_HOME
#   3. W&B login (skipped if ~/.netrc already has a key)
# Each step is skipped if already done, so it is safe to rerun.
#
#   git clone git@github.com:noura3368/command-novelty-rl.git && cd command-novelty-rl
#   ./compute_canada/setup.sh                                  # Qwen2.5-7B and 3B
#   ./compute_canada/setup.sh Qwen/Qwen2.5-1.5B-Instruct       # or name the models
set -euo pipefail
cd "$(dirname "$0")/.."
source compute_canada/env.sh

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
    models=(Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-3B-Instruct)
fi

if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating virtualenv at $VENV"
    virtualenv --no-download "$VENV"
    source "$VENV/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index torch                 # the Alliance wheelhouse build, linked against the cluster's CUDA
    pip install -r requirements.txt              # pinned versions not in the wheelhouse come from PyPI
fi
python -c "import torch, transformers, trl, peft, wandb; print('torch', torch.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__)"

for m in "${models[@]}"; do
    echo "Downloading $m into $HF_HOME"
    python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$m"
done

if grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    echo "W&B key already in ~/.netrc"
else
    wandb login
fi

echo "Done. Jobs should start with: source compute_canada/env.sh"
