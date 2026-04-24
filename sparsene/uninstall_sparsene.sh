#!/bin/bash

set -e

SPARSENE_ROOT=$(pwd)

pip uninstall -y sparsene

rm -rf $SPARSENE_ROOT/examples/src_fp32/acc/testbed/build/
rm -rf $SPARSENE_ROOT/examples/src_fp32/bitbsr/testbed/build/
rm -rf $SPARSENE_ROOT/examples/src_fp32/dtc/testbed/build/
rm -rf $SPARSENE_ROOT/examples/src_fp32/sr_bcrs/testbed/build/
rm -rf $SPARSENE_ROOT/examples/src_fp32/cusparse/testbed/build/