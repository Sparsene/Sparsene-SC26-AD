#!/bin/bash

set -e

SPARSENE_LOADBALANCE_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_LOADBALANCE_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

./build/demo-dtc-tf32 -filename $SPARSENE_AD_ROOT/dataset/selected_mtx/mip1.mtx -N 512 > $SPARSENE_AD_ROOT/results/mip1_no_balance.log
./build/demo-dtc-multi-binding-tf32 -filename $SPARSENE_AD_ROOT/dataset/selected_mtx/mip1.mtx -N 512 > $SPARSENE_AD_ROOT/results/mip1_multi_bind.log

./build/demo-dtc-tf32 -filename $SPARSENE_AD_ROOT/dataset/selected_mtx/ddi.mtx -N 512 > $SPARSENE_AD_ROOT/results/ddi_no_balance.log
./build/demo-dtc-strict-lb-tf32 -filename $SPARSENE_AD_ROOT/dataset/selected_mtx/ddi.mtx -N 512 > $SPARSENE_AD_ROOT/results/ddi_strict_lb.log

