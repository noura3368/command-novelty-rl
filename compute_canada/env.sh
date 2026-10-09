# Shared environment for the Alliance (Compute Canada) clusters. Sourced by setup.sh and by
# every job script, so login-node setup and compute-node runs see the same modules, venv and caches.
#   source compute_canada/env.sh

module load StdEnv/2023 gcc arrow python/3.11   # arrow: pyarrow for `datasets` must come from the module, not pip

export VENV="$HOME/venvs/novelty-rl"
export HF_HOME="$SCRATCH/hf"                    # model weights (~15 GB for 7B); scratch is purged after 60 days unused
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p "$HF_HOME" "$WANDB_DIR"

if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
