#!/usr/bin/env bash
# Checks whether a GPU compute node can reach the sites a W&B sweep needs. Run from the repo root:
#   sbatch --account=def-sfischme compute_canada/test_internet.sh
#   cat test_internet-<jobid>.out
# Nibi needs a GPU type; the smallest H100 slice (10 GB) is enough here and starts soonest.
#SBATCH --job-name=test-internet
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=test_internet-%j.out

check() {
    for url in https://api.wandb.ai https://huggingface.co https://pypi.org; do
        code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 15 "$url" 2>&1) || true
        echo "  $url -> $code"
    done
}

echo "Node: $(hostname)"
echo "Direct:"
check

# Some clusters (e.g. Narval, Rorqual) only allow listed sites through this module.
if module load httpproxy 2>/dev/null; then
    echo "With httpproxy module:"
    check
else
    echo "No httpproxy module on this cluster"
fi

# A real W&B round trip: needs `wandb login` done (setup.sh).
source compute_canada/env.sh
python -c "import wandb; print('W&B user:', wandb.Api(timeout=15).viewer.username)" 2>&1 | tail -1
