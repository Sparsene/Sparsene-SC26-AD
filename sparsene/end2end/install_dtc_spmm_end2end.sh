#!/bin/bash

set -e

SPARSENE_END2END_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_END2END_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

pushd $SPARSENE_AD_ROOT/DTC-SpMM_ASPLOS24
source init_dtc.sh
source third_party/init_sputnik.sh
cd DTC-SpMM
python3 setup.py build_ext --inplace
popd