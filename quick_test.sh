#!/bin/bash

set -e

if [ -t 1 ]; then
	BLUE='\033[1;34m'
	GREEN='\033[1;32m'
	YELLOW='\033[1;33m'
	RED='\033[1;31m'
	NC='\033[0m'
else
	BLUE=''
	GREEN=''
	YELLOW=''
	RED=''
	NC=''
fi

SPARSENE_AD_ROOT=$(pwd)

log_step() { printf "%b[STEP]%b %s\n" "$BLUE" "$NC" "$1"; }
log_ok() { printf "%b[OK]%b %s\n" "$GREEN" "$NC" "$1"; }
log_note() { printf "%b[NOTE]%b %s\n" "$YELLOW" "$NC" "$1"; }
log_err() { printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$1"; }

trap 'log_err "Build failed."' ERR

#! ========================================== Sparsene-AD quick test
log_step "Starting quick test for Sparsene-AD"
pushd $SPARSENE_AD_ROOT/sparsene
./examples/src_fp32/acc/testbed/build/demo-acc-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/bitbsr/testbed/build/demo-bitbsr-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/dtc/testbed/build/demo-dtc-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/dtc/testbed/build/demo-dtc-strict-lb-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/dtc/testbed/build/demo-dtc-multi-binding-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/sr_bcrs/testbed/build/demo-srbcrs-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/sr_bcrs/testbed/build/demo-srbcrs-16x8-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/sr_bcrs/testbed/build/demo-srbcrs-16x8-strict-lb-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
./examples/src_fp32/sr_bcrs/testbed/build/demo-srbcrs-16x8-multi-bind-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
popd
log_ok "Quick test for Sparsene-AD finished."

#! ========================================== AccSpMM quick test
log_step "Starting quick test for AccSpMM"
pushd $SPARSENE_AD_ROOT/AccSpMM
./mma $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx 256
popd
log_ok "AccSpMM quick test finished."

#! ========================================== cuSPARSE quick test
log_step "Starting quick test for cuSPARSE"
pushd $SPARSENE_AD_ROOT/sparsene/examples/src_fp32/cusparse/testbed/build
./demo-cusp-tf32 -filename $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx -N 256
popd
log_ok "cuSPARSE quick test finished."

#! ========================================== Sputnik quick test
log_step "Starting quick test for Sputnik"
pushd $SPARSENE_AD_ROOT/FlashSparse/Baseline/RoDe/build/eval
./eval_spmm_f32_n256 $SPARSENE_AD_ROOT/dataset/cop20k_A.mtx
popd
log_ok "Sputnik quick test finished."


#! ========================================== DTC-SpMM quick test (check dataset availability)
log_step "Starting quick test for DTC-SpMM"
DTCSPMM_NPZ_DIR=$SPARSENE_AD_ROOT/dataset/selected_npz
DTCSPMM_NPZ_FILE=$DTCSPMM_NPZ_DIR/cop20k_A.npz
if [ ! -d "$DTCSPMM_NPZ_DIR" ]; then
    log_err "Missing directory: $DTCSPMM_NPZ_DIR"
    log_note "Please create this directory and convert the dataset using mtx2npz.py."
    exit 1
fi
if [ ! -f "$DTCSPMM_NPZ_FILE" ]; then
    log_err "Missing dataset file: $DTCSPMM_NPZ_FILE"
    log_note "Expected file: dataset/selected_npz/cop20k_A.npz"
    exit 1
fi

pushd $SPARSENE_AD_ROOT/DTC-SpMM_ASPLOS24/scripts/DTCSpMM
python run_DTC_SpMM_selected.py --dataset cop20k_A
log_ok "DTC-SpMM quick test finished."

#! ========================================== FlashSparse quick test (check dataset availability)
log_step "Starting quick test for FlashSparse"
FLASHSPARSE_NPZ_DIR=$SPARSENE_AD_ROOT/dataset/flashsparse_npz
FLASHSPARSE_NPZ_FILE=$FLASHSPARSE_NPZ_DIR/cop20k_A.npz

if [ ! -d "$FLASHSPARSE_NPZ_DIR" ]; then
	log_err "Missing directory: $FLASHSPARSE_NPZ_DIR"
	log_note "Please create this directory and convert the dataset using flashsparse_convert_parallel.py."
	exit 1
fi

if [ ! -f "$FLASHSPARSE_NPZ_FILE" ]; then
	log_err "Missing dataset file: $FLASHSPARSE_NPZ_FILE"
	log_note "Expected file: dataset/flashsparse_npz/cop20k_A.npz"
	exit 1
fi

pushd $SPARSENE_AD_ROOT/FlashSparse/eva/kernel/spmm
python spmm_tf32_run_one.py cop20k_A 121192 2624331 256 1 32 $SPARSENE_AD_ROOT/dataset/flashsparse_npz/ $SPARSENE_AD_ROOT/FlashSparse/result/FlashSparse/spmm/spmm_tf32_256.1.csv
log_ok "FlashSparse quick test finished."

#! ==========================================  SparseTIR quick test
log_step "Starting quick test for SparseTIR"
log_note "Please follow the instructions in the README to set up the environment and run the SparseTIR."
log_ok "SparseTIR quick test finished."