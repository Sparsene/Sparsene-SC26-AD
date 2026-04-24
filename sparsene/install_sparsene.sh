#!/bin/bash

set -e

SPARSENE_ROOT=$(pwd)
# NV_ARCH=${1:?Usage: ./install_sparsene.sh <sm_arch> (e.g. 80, 86, 90a)}

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

log_step() { printf "%b[STEP]%b %s\n" "$BLUE" "$NC" "$1"; }
log_ok() { printf "%b[OK]%b %s\n" "$GREEN" "$NC" "$1"; }
log_note() { printf "%b[NOTE]%b %s\n" "$YELLOW" "$NC" "$1"; }
log_err() { printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$1"; }

trap 'log_err "Installation failed."' ERR

log_step "Starting Sparsene installation in: $SPARSENE_ROOT"
log_note "Make sure your Python and CUDA environment are correctly configured."


cd $SPARSENE_ROOT/python
log_step "Installing Python package (editable mode)"
pip install -e .
log_ok "Python package installed."

# Install sparsene-acc
cd $SPARSENE_ROOT/examples/src_fp32/acc/testbed
log_step "Building sparsene-acc"
mkdir -p build && cd build
cmake .. 
make -j48
log_ok "sparsene-acc build finished."

# Install sparsene-bitbsr
cd $SPARSENE_ROOT/examples/src_fp32/bitbsr/testbed
log_step "Building sparsene-bitbsr"
mkdir -p build && cd build
cmake ..
make -j48
log_ok "sparsene-bitbsr build finished."

# Install sparsene-dtc
cd $SPARSENE_ROOT/examples/src_fp32/dtc/testbed
log_step "Building sparsene-dtc"
mkdir -p build && cd build
cmake ..
make -j48
log_ok "sparsene-dtc build finished."

# Install sparsene-sr_bcrs
cd $SPARSENE_ROOT/examples/src_fp32/sr_bcrs/testbed
log_step "Building sparsene-sr_bcrs"
mkdir -p build && cd build
cmake ..
make -j48
log_ok "sparsene-sr_bcrs build finished."

# Install cusparse baseline
cd $SPARSENE_ROOT/examples/src_fp32/cusparse/testbed
log_step "Building cusparse baseline"
mkdir -p build && cd build
cmake ..
make -j48
log_ok "cusparse baseline build finished."

log_ok "Sparsene installation completed successfully."