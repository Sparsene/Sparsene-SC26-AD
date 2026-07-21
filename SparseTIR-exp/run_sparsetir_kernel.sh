#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"

pushd "$SCRIPT_DIR/examples/spmm" >/dev/null
python exp.py "$@"
popd >/dev/null
