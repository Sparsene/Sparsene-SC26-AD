#!/bin/bash

set -e

SPARSENE_END2END_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_END2END_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

pushd $SPARSENE_AD_ROOT/DTC-SpMM_ASPLOS24
source init_dtc.sh
source third_party/init_sputnik.sh
popd


pushd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | grep -v '^ASIC_680k$' | paste -sd, -)
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=base \
SPARSENE_DTC_SOURCE_ROOT=$SPARSENE_AD_ROOT/DTC-SpMM_ASPLOS24/DTC-SpMM \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=1 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_DTC_TILE_B=auto \
python3 gcn/eva_gcn_sparsene.py \
    --dataset-dir $SPARSENE_AD_ROOT/dataset/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 1 \
    --backend-warmup-iters 2 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend external \
    --external-module external_backend_spmm \
    --external-function dtc_spmm \
    --output-csv result/sc-ad-gcn_e2e_dtc_origin_26_128-512.csv
popd