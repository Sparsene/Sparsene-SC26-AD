#!/bin/bash

set -e

SPARSENE_SEARCH_CONVERGENCE_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_SEARCH_CONVERGENCE_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

pushd $SPARSENE_ROOT/examples/src_fp32/acc/testbed/scripts
python plan_searcher.py
cd ..
mkdir build_test
cd build_test
cmake ..
make -j48
popd