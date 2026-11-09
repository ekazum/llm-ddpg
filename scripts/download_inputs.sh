#!/usr/bin/env bash
set -euo pipefail
# Download the Phase 1 inputs once, from a node with internet access:
# the two UltraFeedback preference parquet files into the lab share, and the
# Qwen3 tokenizer into the shared Hugging Face cache. Links `data` in the repo
# root to the lab directory so config/pipeline.yaml can use relative paths.
#
# Usage:
#   bash scripts/download_inputs.sh
#
# LAB_DIR and HF_HOME can be set in the environment to use other locations.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LAB_DIR=${LAB_DIR:-/private/shaham-lab/ekazum/llm-ddpg}
export HF_HOME=${HF_HOME:-/private/shaham-lab/ekazum/hf_cache}

if [[ -e "$ROOT/data" && ! -L "$ROOT/data" ]]; then
    echo "error: $ROOT/data exists and is not a symlink; move it away first" >&2
    exit 1
fi
mkdir -p "$LAB_DIR/raw" "$LAB_DIR/pipeline" "$HF_HOME"
ln -sfn "$LAB_DIR" "$ROOT/data"
echo "data -> $LAB_DIR"

# Files are named explicitly: `hf download` takes one pattern per --include, and
# positional file names switch the patterns off.
hf download HuggingFaceH4/ultrafeedback_binarized --repo-type dataset \
    data/train_prefs-00000-of-00001.parquet data/test_prefs-00000-of-00001.parquet \
    --local-dir "$ROOT/data/raw/ultrafeedback_binarized"

hf download Qwen/Qwen3-1.7B tokenizer.json tokenizer_config.json vocab.json merges.txt config.json

echo "raw files:"
ls -l "$ROOT/data/raw/ultrafeedback_binarized/data"
