#!/usr/bin/env bash
#SBATCH --job-name=llm-ddpg-baseline
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/baseline_%j.log
set -euo pipefail
# Phase 2 job: the untrained policy scored on dev and on test. Submitted by
# scripts/submit_baseline.sh from the repo root; the first argument is the
# evaluation config.

CONFIG=${1:-config/eval.yaml}
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

python -m src.baselines.untrained.baseline_main --config "$CONFIG"
