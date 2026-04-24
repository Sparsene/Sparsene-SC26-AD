#!/bin/bash

set -e

FLASHSPARSE_ROOT=$(pwd)

if [ -t 1 ]; then
	BLUE='\033[1;34m'
	GREEN='\033[1;32m'
	RED='\033[1;31m'
	NC='\033[0m'
else
	BLUE=''
	GREEN=''
	RED=''
	NC=''
fi

log_step() { printf "%b[STEP]%b %s\n" "$BLUE" "$NC" "$1"; }
log_ok() { printf "%b[OK]%b %s\n" "$GREEN" "$NC" "$1"; }
log_err() { printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$1"; }

trap 'log_err "Installation failed."' ERR

log_step "Installing FlashSparse"
cd $FLASHSPARSE_ROOT/FlashSparse
rm -rf build &&
python setup.py install
log_ok "FlashSparse installation finished."

log_step "Installing Sputnik"
cd $FLASHSPARSE_ROOT/Baseline/RoDe
rm -rf build
mkdir build
cd build
cmake ..
make
log_ok "Sputnik build finished."
