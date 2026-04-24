#!/bin/bash

set -e

FLASHSPARSE_ROOT=$(pwd)

cd $FLASHSPARSE_ROOT/FlashSparse
rm -rf build
pip uninstall -y FlashSparse-kernel

cd $FLASHSPARSE_ROOT/Baseline/RoDe/
rm -rf build