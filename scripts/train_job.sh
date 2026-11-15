#!/usr/bin/env bash
#SBATCH --job-name=llm-ddpg-train
#SBATCH --partition=p_a100_shaham
#SBATCH --account=ug_shaham
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_%j.log
set -euo pipefail
# Training job (critic training, actor frozen). Submitted by
# scripts/submit_train.sh from the repo root; the first argument is the
# training config. One GPU here; submit_train.sh --gpus 2 asks for two.

CONFIG=${1:-config/train.yaml}
CONDA_ENV=${CONDA_ENV:-/private/shaham-lab/ekazum/envs/llm-ddpg}
export HF_HOME=${HF_HOME:-/private/shaham-lab/ekazum/hf_cache}
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Config: $CONFIG  Start: $(date)"
# The end line is printed on every exit, with the exit code.
trap 'echo "End: $(date)  exit=$?"' EXIT

# shellcheck source=/dev/null
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

python -m src.rl_agent.rl_agent_main --config "$CONFIG"
