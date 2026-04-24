#!/bin/bash

set -e

FLASHSPARSE_DIR=$(pwd)
REPO_ROOT=$(cd "$FLASHSPARSE_DIR/.." && pwd)
PY_DIR="$REPO_ROOT/FlashSparse/Baseline/RoDe/script"

DATA_FILTER_FILE="${1:-$REPO_ROOT/dataset/data_filter.csv}"
MTX_DIR="${2:-$REPO_ROOT/dataset/selected_mtx}"

python -u "$PY_DIR/eval_spmm_call_128.py" "$REPO_ROOT" --data-filter "$DATA_FILTER_FILE" --mtx-dir "$MTX_DIR"
python -u "$PY_DIR/eval_spmm_call_256.py" "$REPO_ROOT" --data-filter "$DATA_FILTER_FILE" --mtx-dir "$MTX_DIR"
python -u "$PY_DIR/eval_spmm_call_512.py" "$REPO_ROOT" --data-filter "$DATA_FILTER_FILE" --mtx-dir "$MTX_DIR"