#!/bin/bash

set -e

SPARSENE_END2END_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_END2END_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)



#! ================================================= DTC base (all 26)
pushd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | grep -v '^ASIC_680k$' | paste -sd, -)
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=base \
SPARSENE_DTC_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
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
    --output-csv result/sc-ad-gcn_e2e_dtc_base_26_128-512.csv
popd

#! ================================================= DTC multi-bind (all 26)
pushd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | grep -v '^ASIC_680k$' | paste -sd, -)
SPARSENE_SPMM_BACKEND=dtc \
SPARSENE_DTC_VARIANT=multi_binding \
SPARSENE_DTC_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/dtc/testbed \
SPARSENE_DTC_MODULE_NAME=DTCSpMM \
SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
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
    --output-csv result/sc-ad-gcn_e2e_dtc_multibind_26_128-512.csv
popd

#! ================================================= DTC strict-lb (all 26)
pushd $SPARSENE_END2END_DIR
set -euo pipefail
mapfile -t DATASETS_ARR < <(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | grep -v '^ASIC_680k$')

rm -f result/_tmp_dtc_strict_lb_*.csv result/sc-ad-gcn_e2e_dtc_strict_lb_26.csv

for ds in "${DATASETS_ARR[@]}"; do
    echo "[RUN] ${ds}"
    SPARSENE_SPMM_BACKEND=dtc \
    SPARSENE_DTC_VARIANT=strict_lb \
    SPARSENE_DTC_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/dtc/testbed \
    SPARSENE_DTC_MODULE_NAME=DTCSpMM \
    SPARSENE_DTC_ALLOW_NON_SRC_FP32=0 \
    SPARSENE_ALLOW_CSR_FALLBACK=0 \
    SPARSENE_DTC_TILE_B=auto \
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
            --external-function dtc_spmm \
            --output-csv "result/_tmp_dtc_strict_lb_${ds}.csv"
done

first=1
for ds in "${DATASETS_ARR[@]}"; do
    f="result/_tmp_dtc_strict_lb_${ds}.csv"
    if [[ $first -eq 1 ]]; then
        cat "$f" > result/sc-ad-gcn_e2e_dtc_strict_lb_26_128-512.csv
        first=0
    else
        tail -n +2 "$f" >> result/sc-ad-gcn_e2e_dtc_strict_lb_26_128-512.csv
    fi
done

wc -l result/sc-ad-gcn_e2e_dtc_strict_lb_26_128-512.csv
popd

#! ================================================= SR-BCRS (all 26)
pushd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=base \
SPARSENE_SRBCRS_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
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
    --external-function srbcrs_spmm \
    --output-csv result/sc-ad-gcn_e2e_srbcrs_base_26_128-512.csv
popd

#! ================================================= SR-BCRS 16x8 (all 26)
pushd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8 \
SPARSENE_SRBCRS_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
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
    --external-function srbcrs_spmm \
    --output-csv result/sc-ad-gcn_e2e_srbcrs_16x8_26_128-512.csv
popd

#! ================================================= SR-BCRS 16x8 multi-bind / strict-lb (all 26)
pushd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8_multi_bind \
SPARSENE_SRBCRS_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
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
    --external-function srbcrs_spmm \
    --output-csv result/sc-ad-gcn_e2e_srbcrs_16x8_multibind_26_128-512.csv
popd

pushd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | paste -sd, -)
SPARSENE_SPMM_BACKEND=sr_bcrs \
SPARSENE_SRBCRS_VARIANT=16x8_strict_lb \
SPARSENE_SRBCRS_SOURCE_ROOT=$SPARSENE_ROOT/examples/src_fp32/sr_bcrs/testbed \
SPARSENE_SRBCRS_MODULE_NAME=SRBCRSSpMM \
SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=0 \
SPARSENE_ALLOW_CSR_FALLBACK=0 \
SPARSENE_SRBCRS_TILE_B=auto \
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
    --external-function srbcrs_spmm \
    --output-csv result/sc-ad-gcn_e2e_srbcrs_16x8_strict_lb_26_128-512.csv
popd
