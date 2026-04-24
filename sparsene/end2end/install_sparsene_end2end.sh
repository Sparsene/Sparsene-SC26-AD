#!/bin/bash

set -e

SPARSENE_END2END_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_END2END_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

pushd $SPARSENE_ROOT/examples/src_fp32/dtc/testbed
python3 setup.py build_ext --inplace
popd

pushd $SPARSENE_ROOT/examples/src_fp32/sr_bcrs/testbed
python3 setup.py build_ext --inplace
popd