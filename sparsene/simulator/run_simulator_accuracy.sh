#!/bin/bash
# AE T4: simulator ranking accuracy (Table IV).
#
# Each invocation runs 4 steps (generate plans → profile → compile+time →
# simulate) via the internal runner ./simulator_accuracy.sh, and writes
# results/<gpu>/<format>/combined_results.json. Step 3 is the bottleneck
# (~4–6h per format); see ./simulator_accuracy.sh --help for options like
# --gpus, --compile-jobs, --max-plans (for quick tests).
#
# Prerequisites: CUDA 12.x (nvcc in $PATH), Python 3.8+ with graphviz,
# CUTLASS headers at examples/cutlass/.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
RUNNER="$SCRIPT_DIR/simulator_accuracy.sh"

# -------- A100 (Ampere, sm_80) -------- [default: active]
bash "$RUNNER" --format acc    --arch sm_80 --gpu-name A100 --gpus 0 -o results/a100/acc
bash "$RUNNER" --format bitbsr --arch sm_80 --gpu-name A100 --gpus 0 -o results/a100/bitbsr

# -------- RTX 4090 (Ada, sm_89) -------- [uncomment on RTX 4090 node]
# bash "$RUNNER" --format acc    --arch sm_89 --gpu-name 4090 --gpus 0 -o results/rtx4090/acc
# bash "$RUNNER" --format bitbsr --arch sm_89 --gpu-name 4090 --gpus 0 -o results/rtx4090/bitbsr

# -------- H100 (Hopper, sm_90) -------- [uncomment on H100 node]
# bash "$RUNNER" --format acc    --arch sm_90 --gpu-name H100 --gpus 0 -o results/h100/acc
# bash "$RUNNER" --format bitbsr --arch sm_90 --gpu-name H100 --gpus 0 -o results/h100/bitbsr

echo ""
echo "All enabled (GPU, format) combinations completed."
echo "Results under: $SCRIPT_DIR/results/"
