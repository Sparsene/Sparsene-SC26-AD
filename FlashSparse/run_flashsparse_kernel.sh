#!/bin/bash

set -e

FLASHSPARSE_DIR=$(pwd)
REPO_ROOT=$(cd "$FLASHSPARSE_DIR/.." && pwd)
PY_SCRIPT="$REPO_ROOT/FlashSparse/eva/kernel/spmm/spmm_tf32_test_main.py"

DATA_FILTER_FILE="${1:-$REPO_ROOT/dataset/data_filter.csv}"
MTX_DIR="${2:-$REPO_ROOT/dataset/selected_mtx}"

pushd $FLASHSPARSE_DIR/eva/kernel/spmm
# python -u "$PY_SCRIPT" 128 "$REPO_ROOT" "$DATA_FILTER_FILE" "$MTX_DIR"
python -u "$PY_SCRIPT" 128 "$REPO_ROOT" "$DATA_FILTER_FILE" "$MTX_DIR" &&
python -u "$PY_SCRIPT" 256 "$REPO_ROOT" "$DATA_FILTER_FILE" "$MTX_DIR" &&
python -u "$PY_SCRIPT" 512 "$REPO_ROOT" "$DATA_FILTER_FILE" "$MTX_DIR"
popd