#!/bin/bash

set -e

SPARSENE_DIR=$(pwd)
REPO_ROOT=$(cd "$SPARSENE_DIR/.." && pwd)
PY_SCRIPT="$REPO_ROOT/scripts/test_sparsene_fp32.py"

# python -u "$PY_SCRIPT" 128 "$REPO_ROOT" --impl cusparse

python -u "$PY_SCRIPT" 128 "$REPO_ROOT" --impl cusparse &&
python -u "$PY_SCRIPT" 256 "$REPO_ROOT" --impl cusparse &&
python -u "$PY_SCRIPT" 512 "$REPO_ROOT" --impl cusparse