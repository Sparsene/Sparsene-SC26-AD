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

log_step() { printf "%b[STEP]%b %s\n" "$BLUE" "$NC" "$1"; }
log_ok() { printf "%b[OK]%b %s\n" "$GREEN" "$NC" "$1"; }
log_note() { printf "%b[NOTE]%b %s\n" "$YELLOW" "$NC" "$1"; }
log_err() { printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$1"; }

trap 'log_err "Installation failed."' ERR

log_step "Installing DTC-SpMM dependencies..."
pip install numpy
pip install scipy
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
log_ok "Python dependencies installed."


log_step "Building DTC-SpMM dependencies..."
export DTC_HOME=$(pwd)
source init_dtc.sh
cd third_party/
source ./build_sputnik.sh
log_ok "Sputnik build finished."

log_step "Building DTC-SpMM..."
cd ${DTC_HOME}/DTC-SpMM && source build.sh
log_ok "DTC-SpMM build finished."

log_note "Please manually execute the following commands:"
echo -e "  ${YELLOW}source ./init_dtc.sh${NC}"
echo -e "  ${YELLOW}source ./third_party/init_sputnik.sh${NC}"
log_note "These commands set up environment variables for DTC-SpMM and Sputnik."