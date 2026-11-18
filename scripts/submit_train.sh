#!/usr/bin/env bash
set -euo pipefail
# Training launcher: submits the training job to SLURM. The partition and
# resources are the #SBATCH lines of scripts/train_job.sh.
#
# Usage:
#   bash scripts/submit_train.sh                       # config/train.yaml
#   bash scripts/submit_train.sh --config PATH         # another config
#   bash scripts/submit_train.sh --gpus 2              # two GPUs instead of the job's one;
#                                                      # the config's devices must use them
#   bash scripts/submit_train.sh --partition NAME      # another partition than the job's
#
# The job log is logs/train_<job id>.log.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ARGS=()
SBATCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) ARGS+=("$2"); shift 2 ;;
        --gpus) SBATCH_ARGS+=("--gres=gpu:$2"); shift 2 ;;
        --partition) SBATCH_ARGS+=("--partition=$2"); shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

sbatch "${SBATCH_ARGS[@]}" scripts/train_job.sh "${ARGS[@]}"
