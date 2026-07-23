#!/bin/bash

set -e

SPARSENE_SEARCH_CONVERGENCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_SEARCH_CONVERGENCE_DIR/.." && pwd)

case "${1:-}" in
  ""|--quick-repro)
    TIMING_ARGS=(--quick-repro)
    ;;
  --no-quick-repro)
    TIMING_ARGS=(
      --use-real-timing
      --build-dir "$SPARSENE_ROOT/examples/src_fp32/acc/testbed/build_test"
      --target-template 'acc-plan_{plan_id}-tf32'
      --make-jobs 48
      --run-args '-N 64 -M 1024 -K 1024 -mtx_flag 0'
    )
    ;;
  -h|--help)
    echo "Usage: bash run_search_convergence.sh [--no-quick-repro]"
    echo "  default           Read plan_result.json and sleep 0.5 seconds per profiled plan; no build required."
    echo "  --no-quick-repro  Compile and run plan executables with real timing."
    exit 0
    ;;
  *)
    echo "Error: unknown option: $1" >&2
    echo "Usage: bash run_search_convergence.sh [--no-quick-repro]" >&2
    exit 2
    ;;
esac

if (( $# > 1 )); then
  echo "Error: too many arguments" >&2
  echo "Usage: bash run_search_convergence.sh [--no-quick-repro]" >&2
  exit 2
fi

pushd "$SPARSENE_ROOT/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans"
python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results sim_scores_6156.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source sim \
  --bad-start-ramp-steps 24 \
  --threshold 0.20 \
  --cluster-cache-file cache/cluster_6156_t020_a4_b2_g1.json \
  "${TIMING_ARGS[@]}" \
  --output-json results/run_sim_cluster_hybrid.json \
  --output-csv results/run_sim_cluster_hybrid.csv \
  --output-plot results/run_sim_cluster_hybrid.png

python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results sim_scores_6156_const.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source distance \
  --bad-start-ramp-steps 24 \
  --threshold 0.20 \
  --cluster-cache-file cache/cluster_6156_t020_a4_b2_g1.json \
  "${TIMING_ARGS[@]}" \
  --output-json results/run_sim_cluster_hybrid_constsim.json \
  --output-csv results/run_sim_cluster_hybrid_constsim.csv \
  --output-plot results/run_sim_cluster_hybrid_constsim.png

python3 plot_sim_vs_nosim_curve.py \
  --sim-json results/run_sim_cluster_hybrid.json \
  --nosim-json results/run_sim_cluster_hybrid_constsim.json \
  --sim-label sim \
  --nosim-label no-sim \
  --output-plot results/plot_sim_vs_nosim_curve.png \
  --output-csv results/plot_sim_vs_nosim_curve.csv

popd
