#!/bin/bash

set -e

SPARSENE_END2END_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_END2END_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

pushd $SPARSENE_END2END_DIR
set -euo pipefail
mapfile -t DATASETS_ARR < <(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt)

rm -f result/_tmp_flashsparse_*.csv result/sc-ad-gcn_e2e_flashsparse_26_128-512.csv

for ds in "${DATASETS_ARR[@]}"; do
    echo "[RUN] ${ds}"
    SPARSENE_SPMM_BACKEND=flashsparse \
    SPARSENE_FLASHSPARSE_VARIANT=tf32_balance \
    SPARSENE_FLASHSPARSE_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/flashsparse/testbed \
    SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT=1 \
    python3 gcn/eva_gcn_sparsene.py \
            --dataset-dir $SPARSENE_AD_ROOT/dataset/selected_npz \
            --datasets "${ds}" \
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
            --external-function flashsparse_spmm \
            --output-csv "result/_tmp_flashsparse_${ds}.csv"
done

first=1
for ds in "${DATASETS_ARR[@]}"; do
    f="result/_tmp_flashsparse_${ds}.csv"
    if [[ $first -eq 1 ]]; then
        cat "$f" > result/sc-ad-gcn_e2e_flashsparse_26_128-512.csv
        first=0
    else
        tail -n +2 "$f" >> result/sc-ad-gcn_e2e_flashsparse_26_128-512.csv
    fi
done

wc -l result/sc-ad-gcn_e2e_flashsparse_26_128-512.csv