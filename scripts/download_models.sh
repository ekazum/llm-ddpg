#!/usr/bin/env bash
set -euo pipefail
# Download the Phase 2 models once, from a node with internet access: the
# policy and the reward model into the shared Hugging Face cache. The jobs
# read them offline.
#
# Usage:
#   bash scripts/download_models.sh
#
# HF_HOME can be set in the environment to use another cache.

export HF_HOME=${HF_HOME:-/private/shaham-lab/ekazum/hf_cache}
mkdir -p "$HF_HOME"

hf download Qwen/Qwen3-1.7B
hf download Skywork/Skywork-Reward-V2-Qwen3-1.7B

echo "models in $HF_HOME/hub:"
ls -d "$HF_HOME"/hub/models--Qwen--Qwen3-1.7B "$HF_HOME"/hub/models--Skywork--Skywork-Reward-V2-Qwen3-1.7B
