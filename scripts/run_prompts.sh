#!/usr/bin/env bash
set -euo pipefail
# Phase 1 launcher: runs the prompt pipeline on this node's CPUs through
# Snakemake, or prints the plan with --dry-run. No SLURM.
#
# Usage:
#   bash scripts/run_prompts.sh                # run with 4 cores
#   bash scripts/run_prompts.sh --cores 8      # run with 8 cores
#   bash scripts/run_prompts.sh --dry-run      # print the plan, run nothing
#
# A rerun continues from the first missing output. The log is logs/prompts_<timestamp>.log.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

TARGET="prompts"
CONDA_ENV=${CONDA_ENV:-/private/shaham-lab/ekazum/envs/llm-ddpg}
CORES=4
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --cores) CORES="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# shellcheck source=/dev/null
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if $DRY_RUN; then
    snakemake --snakefile pipeline/Snakefile --cores "$CORES" --dry-run "$TARGET"
    exit 0
fi

LOG="logs/prompts_$(date +%Y%m%d_%H%M%S).log"
echo "Host: $(hostname)  Cores: $CORES  Start: $(date)  Log: $LOG"
snakemake --snakefile pipeline/Snakefile --cores "$CORES" --rerun-incomplete --printshellcmds "$TARGET" 2>&1 | tee "$LOG"
echo "End: $(date)"
