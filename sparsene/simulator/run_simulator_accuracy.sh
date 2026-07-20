#!/bin/bash
# AE T4: simulator ranking accuracy (Table IV).
#
# Each invocation runs 4 steps (generate plans → profile → compile+time →
# simulate) via the internal runner ./simulator_accuracy.sh. For artifact
# reproduction, Step 3 measures a deterministic sample and fills the remaining
# timings from the precomputed full-run result.
#
# Prerequisites: CUDA 12.x (nvcc in $PATH), Python 3.8+, and CUTLASS headers
# in the testbed directories.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
RUNNER="$SCRIPT_DIR/simulator_accuracy.sh"

# -------- A100 (Ampere, sm_80) -------- [default: active]
SAMPLE_DIVISOR=10
SAMPLE_SEED=2026
A100_PRECOMPUTED="$SCRIPT_DIR/results/a100"
A100_OUTPUT="$SCRIPT_DIR/results/a100_sampled"

bash "$RUNNER" --format acc --arch sm_80 --gpu-name A100 --gpus 0 \
    --sample-divisor "$SAMPLE_DIVISOR" --sample-seed "$SAMPLE_SEED" \
    --precomputed-results "$A100_PRECOMPUTED/acc/precomputed_results.json" \
    -o "$A100_OUTPUT/acc"
bash "$RUNNER" --format bitbsr --arch sm_80 --gpu-name A100 --gpus 0 \
    --sample-divisor "$SAMPLE_DIVISOR" --sample-seed "$SAMPLE_SEED" \
    --precomputed-results "$A100_PRECOMPUTED/bitbsr/precomputed_results.json" \
    -o "$A100_OUTPUT/bitbsr"

# -------- RTX 4090 (Ada, sm_89) -------- [uncomment on RTX 4090 node]
# bash "$RUNNER" --format acc    --arch sm_89 --gpu-name 4090 --gpus 0 -o "$SCRIPT_DIR/results/rtx4090/acc"
# bash "$RUNNER" --format bitbsr --arch sm_89 --gpu-name 4090 --gpus 0 -o "$SCRIPT_DIR/results/rtx4090/bitbsr"

# -------- H100 (Hopper, sm_90) -------- [uncomment on H100 node]
# bash "$RUNNER" --format acc    --arch sm_90 --gpu-name H100 --gpus 0 -o "$SCRIPT_DIR/results/h100/acc"
# bash "$RUNNER" --format bitbsr --arch sm_90 --gpu-name H100 --gpus 0 -o "$SCRIPT_DIR/results/h100/bitbsr"

echo ""
echo "All enabled (GPU, format) combinations completed."
echo "Sampled reproduction results under: $A100_OUTPUT/"
