#!/bin/bash

set -e

SPARSENE_END2END_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_END2END_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

cd $SPARSENE_END2END_DIR
DATASETS=$(sed -E 's#.*/##; s#\.mtx$##' $SPARSENE_AD_ROOT/dataset/filtered_mtx.txt | paste -sd, -)
# $SPARSENE_AD_ROOT/venv_dgl/bin/python gcn/eva_gcn_sparsene.py \
/workspace/venv_dgl/bin/python gcn/eva_gcn_sparsene.py \
    --dataset-dir $SPARSENE_AD_ROOT/dataset/selected_npz \
    --datasets "$DATASETS" \
    --hidden-list 128 \
    --layer-list 3 \
    --epochs 100 \
    --warmup-epochs 30 \
    --repeat-runs 2 \
    --segment-timing 1 \
    --exclude-preprocess 0 \
    --backend-warmup-iters 0 \
    --feature-dim 128 \
    --num-classes 16 \
    --device cuda:0 \
    --backend dgl \
    --output-csv result/sc-ad-gcn_e2e_dgl_26_128-512.csv