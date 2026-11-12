#!/usr/bin/env bash
set -euo pipefail
# Phase 2 launcher: submits the baseline job (the untrained policy scored on
# dev and on test) to SLURM. The partition, account and resources are the
# #SBATCH lines of scripts/baseline_job.sh.
#
# Usage:
#   bash scripts/submit_baseline.sh                       # config/eval.yaml
#   bash scripts/submit_baseline.sh --config PATH         # another config
#
# The job log is logs/baseline_<job id>.log.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) ARGS+=("$2"); shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

sbatch scripts/baseline_job.sh "${ARGS[@]}"
