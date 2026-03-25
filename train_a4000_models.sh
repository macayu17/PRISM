#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" check_a4000_ready.py
"$PYTHON_BIN" src/train_model_suite.py train \
  --run-name a4000_full \
  --gpu-profile rtx-a4000 \
  --epochs 30 \
  --patience 8 \
  --traditional-trials 6 \
  --transformer-trials 6 \
  "$@"
