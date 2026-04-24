#!/bin/bash

set -e

ACCSPMM_ROOT=$(pwd)

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

trap 'log_err "Build failed."' ERR

log_step "Starting AccSpMM build in: $ACCSPMM_ROOT"

mkdir -p build && cd build
cmake .. 
make -j48

log_ok "AccSpMM build finished."