#!/usr/bin/env bash
set -euo pipefail
# Training launcher: submits the training job to SLURM. The partition and
# resources are the #SBATCH lines of scripts/train_job.sh.
#
# Usage:
#   bash scripts/submit_train.sh                       # config/train.yaml
#   bash scripts/submit_train.sh --config PATH         # another config
#
# The job log is logs/train_<job id>.log.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) ARGS+=("$2"); shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

sbatch scripts/train_job.sh "${ARGS[@]}"
