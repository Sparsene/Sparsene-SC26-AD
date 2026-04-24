#!/bin/bash

set -e

SPARSENE_AD_ROOT=$(pwd)

#! T1: kernels
pushd $SPARSENE_AD_ROOT/scripts
# sparsene
python process_sparsene_test_log.py ../results/sparsene_fp32_N128.log ../results/jsons/sparsene_fp32_N128.json fp32
python process_sparsene_test_log.py ../results/sparsene_fp32_N256.log ../results/jsons/sparsene_fp32_N256.json fp32
python process_sparsene_test_log.py ../results/sparsene_fp32_N512.log ../results/jsons/sparsene_fp32_N512.json fp32
# flashsparse
python process_flashsparse_test_csv.py -N 128 ../FlashSparse/result/FlashSparse/spmm/spmm_tf32_128.csv ../results/jsons/flashsparse_fp32_N128.json
python process_flashsparse_test_csv.py -N 256 ../FlashSparse/result/FlashSparse/spmm/spmm_tf32_256.csv ../results/jsons/flashsparse_fp32_N256.json
python process_flashsparse_test_csv.py -N 512 ../FlashSparse/result/FlashSparse/spmm/spmm_tf32_512.csv ../results/jsons/flashsparse_fp32_N512.json
# acc-spmm
python process_acc_test_csv.py ../results/acc-result.csv ../results/jsons/acc_fp32.json
# dtc-spmm
python process_dtc_test_csv.py ../results/DTCSpMM_exe_time_and_throughput.csv ../results/jsons/dtc_fp32.json
# cusparse
python process_cusparse_test_log.py ../results/cusparse_fp32_N128.log ../results/jsons/cusparse_fp32_N128.json fp32
python process_cusparse_test_log.py ../results/cusparse_fp32_N256.log ../results/jsons/cusparse_fp32_N256.json fp32
python process_cusparse_test_log.py ../results/cusparse_fp32_N512.log ../results/jsons/cusparse_fp32_N512.json fp32
# sparsetir
python process_sparsetir_test_log.py ../SparseTIR-exp/examples/spmm/sparsetir.log ../results/jsons/
# sputnik
python process_sputnik_test_csv.py ../FlashSparse/result/Baseline/spmm/rode_spmm_f32_n128.csv ../results/jsons/sputnik_fp32_N128.json 128
python process_sputnik_test_csv.py ../FlashSparse/result/Baseline/spmm/rode_spmm_f32_n256.csv ../results/jsons/sputnik_fp32_N256.json 256
python process_sputnik_test_csv.py ../FlashSparse/result/Baseline/spmm/rode_spmm_f32_n512.csv ../results/jsons/sputnik_fp32_N512.json 512

# merge all json files
python merge_kernel_json.py -i ../results/jsons/ -o ../results/jsons/
# plot fig7
python plot_fig7_single_case_speedup.py

# generate table III
python plot_table3_kernel.py

#! T2: end2end
pushd $SPARSENE_AD_ROOT/scripts
# generate table V
python plot_table5_end2end.py -d ../sparsene/end2end/result
popd

#! T3: load balance
pushd $SPARSENE_AD_ROOT/sparsene/load_balance
# plot fig13
python plot_sm_balance_compare_new.py --matrices mip1,ddi --without-logs ../../results/mip1_no_balance.log,../../results/ddi_no_balance.log --with-logs ../../results/mip1_multi_bind.log,../../results/ddi_strict_lb.log --without-labels "w/o balance" --with-labels "w/ multi-bind,w/ strict lb" --output mip1_ddi_sm_balance_compare.png
popd

#! T4: pipeline simulator
pushd $SPARSENE_AD_ROOT/sparsene/simulator
# generate table IV
python eval_topk_precision.py --json ./results/a100/acc/combined_results.json
python eval_topk_precision.py --json ./results/a100/bitbsr/combined_results.json
popd

#! T5: search convergence
pushd $SPARSENE_AD_ROOT/sparsene/search_convergence
# plot fig14
python plot_sim_vs_nosim_curve.py \
  --sim-json results/run_sim_cluster_hybrid.json \
  --nosim-json results/run_sim_cluster_hybrid_constsim.json \
  --sim-label sim \
  --nosim-label no-sim \
  --output-plot results/plot_sim_vs_nosim_curve.png \
  --output-csv results/plot_sim_vs_nosim_curve.csv
popd