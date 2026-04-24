/**
 * host_program_profiling.cu — Sequential per-op profiling
 * ========================================================
 *
 * Drop-in replacement for host_program.cu. Reuse with main.cu:
 *   main.cu + host_program_profiling.cu → profiling binary
 *
 * Instead of the pipelined kernel, launches a sequential profiling kernel
 * that executes ops one at a time with clock() timing. Produces correct
 * output (verified via cusparse in main.cu) plus per-op T_issue data.
 *
 * Output format (stdout):
 *   PROFILE <iteration> <op_id> <op_name> <t_issue>
 *   mykernel_time: <ms>   (for compatibility with existing parsing)
 */

#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "kernel.inc"

using namespace cute;
using namespace std;

// ---------------------------------------------------------------------------
// Profiling data structures
// ---------------------------------------------------------------------------

struct ProfileEntry {
    uint32_t op_id;
    uint32_t t_issue;
};

#define NUM_OPS 9
#define MAX_PROFILE_ENTRIES 2048  // 9 ops × up to ~220 iterations (fp32: Tile_K=8, more tiles)

__device__ __forceinline__ uint32_t prof_clock() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(r));
    return r;
}

// Busy-wait for N clock cycles. Ensures all async ops (ld.global, cp.async, mma)
// from the previous op have completed before the next op's clock measurement.
// The sleep is OUTSIDE the clock measurement so it doesn't affect T_issue.
#define PROF_DRAIN_CYCLES 500
__device__ __forceinline__ void prof_drain() {
    uint32_t start = prof_clock();
    while (prof_clock() - start < PROF_DRAIN_CYCLES) {}
}

// Store profiling entry to shared memory via inline PTX (same technique as trace)
__device__ __forceinline__ void prof_store_smem(ProfileEntry* buf, int idx,
                                                 uint32_t op_id, uint32_t t_issue) {
    buf[idx].op_id = op_id;
    buf[idx].t_issue = t_issue;
}

// ---------------------------------------------------------------------------
// Sequential profiling kernel
// ---------------------------------------------------------------------------
// Executes all 9 ops one at a time per iteration, with clock() timing.
// Uses NBUF=1 (single buffer) — no multi-buffering needed.
// All blocks compute correct results; only profiling blocks record timing.

template <class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK, class WARP_MNK,
          class nmask_per_tile, class tile_mco_val_size>
__global__ void bitbsr_spmm_profiling_kernel(
    float *dB_val,
    float *dC_val,
    uint64_t *dval_mco_mask,
    int *dval_mco_off,
    float *dval_mco_val,
    int *dval_sidx,
    int *dval_soff,
    int K, int M, int Mo, int N, int nnz, int nnz_block,
    ProfileEntry *d_profile, int *d_profile_count,
    int prof_mode,   // 0 = T_issue, 1 = T_tail sandwich
    int prof_nblocks // number of blocks to profile (0 = none)
) {
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int wid = tid / 32;

    // Use NBUF=1 for single-buffered sequential execution
    using NBUF = Int<1>;

    // ---- Instantiate input tensor ops (same as pipelined kernel) ----
    using B_valOp_t = B_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    B_valOp_t B_valOp_v{tid, lid, wid, K, N, dB_val};
    using C_valOp_t = C_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    C_valOp_t C_valOp_v{tid, lid, wid, M, N, dC_val};
    using val_mco_maskOp_t = val_mco_maskOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_mco_maskOp_t val_mco_maskOp_v{tid, lid, wid, nnz_block, dval_mco_mask};
    using val_mco_offOp_t = val_mco_offOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_mco_offOp_t val_mco_offOp_v{tid, lid, wid, nnz_block, dval_mco_off};
    using val_mco_valOp_t = val_mco_valOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_mco_valOp_t val_mco_valOp_v{tid, lid, wid, nnz, dval_mco_val};
    using val_sidxOp_t = val_sidxOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_sidxOp_t val_sidxOp_v{tid, lid, wid, nnz_block, dval_sidx};
    using val_soffOp_t = val_soffOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    val_soffOp_t val_soffOp_v{tid, lid, wid, Mo, dval_soff};

    // ---- Scalar/setup ops ----
    using ZeroOp_t = ZeroOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZeroOp_layout_o0 = typename ZeroOp_t::Layout_o0;
    int ZeroOp_tensor_o0[cosize_v<ZeroOp_layout_o0>];
    ZeroOp_t ZeroOp_v{tid, lid, wid, ZeroOp_tensor_o0};

    using BlockDimYOp_t = BlockDimYOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimYOp_layout_o0 = typename BlockDimYOp_t::Layout_o0;
    int BlockDimYOp_tensor_o0[cosize_v<BlockDimYOp_layout_o0>];
    BlockDimYOp_t BlockDimYOp_v{tid, lid, wid, BlockDimYOp_tensor_o0};

    using BlockDimXOp_t = BlockDimXOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using BlockDimXOp_layout_o0 = typename BlockDimXOp_t::Layout_o0;
    int BlockDimXOp_tensor_o0[cosize_v<BlockDimXOp_layout_o0>];
    BlockDimXOp_t BlockDimXOp_v{tid, lid, wid, BlockDimXOp_tensor_o0};

    using G2rSparseOffsetLoadOp_t = G2rSparseOffsetLoadOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseOffsetLoadOp_layout_o0 = typename G2rSparseOffsetLoadOp_t::Layout_o0;
    using G2rSparseOffsetLoadOp_layout_o1 = typename G2rSparseOffsetLoadOp_t::Layout_o1;
    int G2rSparseOffsetLoadOp_tensor_o0[cosize_v<G2rSparseOffsetLoadOp_layout_o0>];
    int G2rSparseOffsetLoadOp_tensor_o1[cosize_v<G2rSparseOffsetLoadOp_layout_o1>];
    G2rSparseOffsetLoadOp_t G2rSparseOffsetLoadOp_v{tid, lid, wid,
        G2rSparseOffsetLoadOp_tensor_o0, G2rSparseOffsetLoadOp_tensor_o1, Mo};

    using ZerosOp_t = ZerosOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using ZerosOp_layout_o0 = typename ZerosOp_t::Layout_o0;
    float ZerosOp_tensor_o0[cosize_v<ZerosOp_layout_o0>];
    ZerosOp_t ZerosOp_v{tid, lid, wid, ZerosOp_tensor_o0};

    // ---- Pipeline ops with NBUF=1 (single buffer) ----
    // Op 0: G2rSparseIndexLoadOp — ld.global → register (NOT smem)
    using G2rSparseIndexLoadOp_t = G2rSparseIndexLoadOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseIndexLoadOp_layout_o0 = typename G2rSparseIndexLoadOp_t::Layout_o0;
    int G2rSparseIndexLoadOp_tensor_o0[cosize_v<G2rSparseIndexLoadOp_layout_o0>];
    G2rSparseIndexLoadOp_t G2rSparseIndexLoadOp_v{tid, lid, wid,
        G2rSparseIndexLoadOp_tensor_o0, nnz_block};

    using G2rSparseMcoOffLoadOp_t = G2rSparseMcoOffLoadOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseMcoOffLoadOp_layout_o0 = typename G2rSparseMcoOffLoadOp_t::Layout_o0;
    int G2rSparseMcoOffLoadOp_tensor_o0[cosize_v<G2rSparseMcoOffLoadOp_layout_o0>];
    G2rSparseMcoOffLoadOp_t G2rSparseMcoOffLoadOp_v{tid, lid, wid,
        G2rSparseMcoOffLoadOp_tensor_o0, nnz_block};

    using G2rSparseMcoMaskLoadOp_t = G2rSparseMcoMaskLoadOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2rSparseMcoMaskLoadOp_layout_o0 = typename G2rSparseMcoMaskLoadOp_t::Layout_o0;
    uint64_t G2rSparseMcoMaskLoadOp_tensor_o0[cosize_v<G2rSparseMcoMaskLoadOp_layout_o0>];
    G2rSparseMcoMaskLoadOp_t G2rSparseMcoMaskLoadOp_v{tid, lid, wid,
        G2rSparseMcoMaskLoadOp_tensor_o0, nnz_block};

    using G2sSparseMcoValLoadOp_t = G2sSparseMcoValLoadOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sSparseMcoValLoadOp_layout_o0 = typename G2sSparseMcoValLoadOp_t::Layout_o0;
    __shared__ float G2sSparseMcoValLoadOp_tensor_o0[cosize_v<G2sSparseMcoValLoadOp_layout_o0>];
    G2sSparseMcoValLoadOp_t G2sSparseMcoValLoadOp_v{tid, lid, wid,
        G2sSparseMcoValLoadOp_tensor_o0, nnz};

    using G2sMatrixBLoadOp_t = G2sMatrixBLoadOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using G2sMatrixBLoadOp_layout_o0 = typename G2sMatrixBLoadOp_t::Layout_o0;
    __shared__ float G2sMatrixBLoadOp_tensor_o0[cosize_v<G2sMatrixBLoadOp_layout_o0>];
    G2sMatrixBLoadOp_t G2sMatrixBLoadOp_v{tid, lid, wid,
        G2sMatrixBLoadOp_tensor_o0, K, N};

    using S2sRestoreMatrixAOp_t = S2sRestoreMatrixAOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2sRestoreMatrixAOp_layout_o0 = typename S2sRestoreMatrixAOp_t::Layout_o0;
    __shared__ float S2sRestoreMatrixAOp_tensor_o0[cosize_v<S2sRestoreMatrixAOp_layout_o0>];
    S2sRestoreMatrixAOp_t S2sRestoreMatrixAOp_v{tid, lid, wid,
        S2sRestoreMatrixAOp_tensor_o0};

    using S2rAValLoadOp_t = S2rAValLoadOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rAValLoadOp_layout_o0 = typename S2rAValLoadOp_t::Layout_o0;
    float S2rAValLoadOp_tensor_o0[cosize_v<S2rAValLoadOp_layout_o0>];
    S2rAValLoadOp_t S2rAValLoadOp_v{tid, lid, wid, S2rAValLoadOp_tensor_o0};

    using S2rBValLoadOp_t = S2rBValLoadOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK>;
    using S2rBValLoadOp_layout_o0 = typename S2rBValLoadOp_t::Layout_o0;
    float S2rBValLoadOp_tensor_o0[cosize_v<S2rBValLoadOp_layout_o0>];
    S2rBValLoadOp_t S2rBValLoadOp_v{tid, lid, wid, S2rBValLoadOp_tensor_o0};

    using CalculateOp_t = CalculateOp<NBUF, BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                       std::decay_t<decltype(ZerosOp_v.template output<0>())>>;
    CalculateOp_t CalculateOp_v{tid, lid, wid, ZerosOp_v.template output<0>()};

    // ---- Profiling buffer in shared memory (flush to GMEM at end) ----
    __shared__ ProfileEntry _prof_buf[MAX_PROFILE_ENTRIES];
    int prof_idx = 0;
    int block_linear_idx = blockIdx.x * gridDim.y + blockIdx.y;
    bool is_profiling_block = (block_linear_idx < prof_nblocks);

    // ---- Setup (same as pipelined kernel) ----
    ZeroOp_v.template f<>();
    BlockDimYOp_v.template f<>();
    BlockDimXOp_v.template f<>();
    G2rSparseOffsetLoadOp_v.template f<>(
        val_soffOp_v.template output<0>(),
        blockIdx.x
    );
    ZerosOp_v.template f<>();

    int l = G2rSparseOffsetLoadOp_v.template output<0>()(0);
    int r = G2rSparseOffsetLoadOp_v.template output<1>()(0);
    int k = r - l;

    // Pre-create R2gCValStoreOp for sandwich 8 and final store
    using R2gCValStoreOp_t = R2gCValStoreOp<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                              std::decay_t<decltype(C_valOp_v.template output<0>())>>;
    R2gCValStoreOp_t R2gCValStoreOp_v{tid, lid, wid, C_valOp_v.template output<0>()};

    // ---- Mode 0: T_issue (sequential, each op isolated with drain) ----
    // ---- Mode 1: T_tail sandwich (producer+consumer back-to-back) ----
    for (int iter = 0; iter < k; iter++) {
        uint32_t t0, t1;

      if (prof_mode == 0) {
        // === T_issue measurement ===
        // Each op: clock → op → clock → store → drain(500 cycles)

        // Op 0: G2rSparseIndexLoadOp (ld.global sidx → register)
        t0 = prof_clock();
        G2rSparseIndexLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(), l + iter);
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 0, t1 - t0);
        prof_drain();

        // Op 1: G2rSparseMcoOffLoadOp (ld.global mco_off → reg)
        t0 = prof_clock();
        G2rSparseMcoOffLoadOp_v.template f<0>(
            val_mco_offOp_v.template output<0>(), l + iter);
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 1, t1 - t0);
        prof_drain();

        // Op 2: G2rSparseMcoMaskLoadOp (ld.global mco_mask → reg)
        t0 = prof_clock();
        G2rSparseMcoMaskLoadOp_v.template f<0>(
            val_mco_maskOp_v.template output<0>(), l + iter);
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 2, t1 - t0);
        prof_drain();

        // Op 3: G2sSparseMcoValLoadOp (cp.async mco_val → smem)
        t0 = prof_clock();
        G2sSparseMcoValLoadOp_v.template f<0>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 0>(), l + iter);
        t1 = prof_clock();
        __pipeline_commit();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 3, t1 - t0);
        __pipeline_wait_prior(0); __syncthreads(); prof_drain();

        // Op 4: G2sMatrixBLoadOp (indirect cp.async B → smem)
        t0 = prof_clock();
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 0>()(0));
        t1 = prof_clock();
        __pipeline_commit();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 4, t1 - t0);
        __pipeline_wait_prior(0); __syncthreads(); prof_drain();

        // Op 5: S2sRestoreMatrixAOp (popcll scatter → smem)
        t0 = prof_clock();
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 5, t1 - t0);
        __syncthreads(); prof_drain();

        // Op 6: S2rAValLoadOp (ldmatrix A → reg)
        t0 = prof_clock();
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 6, t1 - t0);
        prof_drain();

        // Op 7: S2rBValLoadOp (ldmatrix B → reg)
        t0 = prof_clock();
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 7, t1 - t0);
        prof_drain();

        // Op 8: CalculateOp (mma.sync)
        t0 = prof_clock();
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 8, t1 - t0);
        prof_drain();

      } else {
        // === T_tail sandwich measurement ===
        // For each pair: drain → clock → opA → (sync if needed) → opB → clock → drain
        // op_id stored = producer op_id (0-8), value = total sandwich time

        // Sandwich 0: Op0(G2rSparseIndexLoadOp) → Op4(G2sMatrixBLoadOp)
        // Op0 is ld.global → register, no pipeline commit needed
        prof_drain();
        t0 = prof_clock();
        G2rSparseIndexLoadOp_v.template f<0>(
            val_sidxOp_v.template output<0>(), l + iter);
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 0>()(0));
        t1 = prof_clock();
        __pipeline_commit();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 0, t1 - t0);
        __pipeline_wait_prior(0); __syncthreads(); prof_drain();

        // Sandwich 1: Op1(G2rSparseMcoOffLoadOp) → Op3(G2sSparseMcoValLoadOp)
        // Direct: no other deps for Op3
        prof_drain();
        t0 = prof_clock();
        G2rSparseMcoOffLoadOp_v.template f<0>(
            val_mco_offOp_v.template output<0>(), l + iter);
        G2sSparseMcoValLoadOp_v.template f<0>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 0>(), l + iter);
        t1 = prof_clock();
        __pipeline_commit();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 1, t1 - t0);
        __pipeline_wait_prior(0); __syncthreads(); prof_drain();

        // Sandwich 2: Op2(G2rSparseMcoMaskLoadOp) → Op5(S2sRestoreMatrixAOp)
        // Op5 also needs Op3 output → pre-run Op3 (already in smem from sandwich 1)
        // smem already has mco_val from sandwich 1's drain, so Op5 can read it
        prof_drain();
        t0 = prof_clock();
        G2rSparseMcoMaskLoadOp_v.template f<0>(
            val_mco_maskOp_v.template output<0>(), l + iter);
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 2, t1 - t0);
        __syncthreads(); prof_drain();

        // Sandwich 3: Op3(G2sSparseMcoValLoadOp) → Op5(S2sRestoreMatrixAOp)
        // Op5 also needs Op2 output → pre-run Op2
        G2rSparseMcoMaskLoadOp_v.template f<0>(
            val_mco_maskOp_v.template output<0>(), l + iter);
        prof_drain();
        t0 = prof_clock();
        G2sSparseMcoValLoadOp_v.template f<0>(
            val_mco_valOp_v.template output<0>(),
            G2rSparseMcoOffLoadOp_v.template output<0, 0>(), l + iter);
        __pipeline_commit();
        __pipeline_wait_prior(0); __syncthreads();
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 3, t1 - t0);
        __syncthreads(); prof_drain();

        // Sandwich 4: Op4(G2sMatrixBLoadOp) → Op7(S2rBValLoadOp)
        // Direct: no other deps for Op7
        // Need sidx in register first (from sandwich 0, already there)
        prof_drain();
        t0 = prof_clock();
        G2sMatrixBLoadOp_v.template f<0>(
            B_valOp_v.template output<0>(),
            G2rSparseIndexLoadOp_v.template output<0, 0>()(0));
        __pipeline_commit();
        __pipeline_wait_prior(0); __syncthreads();
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 4, t1 - t0);
        prof_drain();

        // Sandwich 5: Op5(S2sRestoreMatrixAOp) → Op6(S2rAValLoadOp)
        // Direct: no other deps for Op6
        // smem has mco_val and mask from earlier sandwiches
        prof_drain();
        t0 = prof_clock();
        S2sRestoreMatrixAOp_v.template f<0>(
            G2sSparseMcoValLoadOp_v.template output<0, 0>(),
            G2rSparseMcoMaskLoadOp_v.template output<0, 0>());
        __syncthreads();
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 5, t1 - t0);
        prof_drain();

        // Sandwich 6: Op6(S2rAValLoadOp) → Op8(CalculateOp)
        // Op8 also needs Op7 → pre-run Op7
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>());
        prof_drain();
        t0 = prof_clock();
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>());
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 6, t1 - t0);
        prof_drain();

        // Sandwich 7: Op7(S2rBValLoadOp) → Op8(CalculateOp)
        // Op8 also needs Op6 → pre-run Op6
        S2rAValLoadOp_v.template f<0>(
            S2sRestoreMatrixAOp_v.template output<0, 0>());
        prof_drain();
        t0 = prof_clock();
        S2rBValLoadOp_v.template f<0>(
            G2sMatrixBLoadOp_v.template output<0, 0>());
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 7, t1 - t0);
        prof_drain();

        // Sandwich 8: Op8(CalculateOp) → R2gCValStoreOp
        // Direct: store reads accumulator registers
        prof_drain();
        t0 = prof_clock();
        CalculateOp_v.template f<0>(
            S2rAValLoadOp_v.template output<0, 0>(),
            S2rBValLoadOp_v.template output<0, 0>());
        R2gCValStoreOp_v.f(CalculateOp_v.template output<0, 0>());
        t1 = prof_clock();
        if (is_profiling_block && prof_idx < MAX_PROFILE_ENTRIES) prof_store_smem(_prof_buf, prof_idx++, 8, t1 - t0);
        prof_drain();
      }
    }

    // ---- Store result (same as pipelined kernel) ----
    R2gCValStoreOp_v.f(CalculateOp_v.template output<0, 0>());

    // ---- Flush profiling data from smem to GMEM ----
    if (is_profiling_block && d_profile != nullptr) {
        ProfileEntry* dst = d_profile + block_linear_idx * MAX_PROFILE_ENTRIES;
        for (int _i = 0; _i < prof_idx; _i++) {
            dst[_i] = _prof_buf[_i];
        }
        d_profile_count[block_linear_idx] = prof_idx;
    }
}

// ---------------------------------------------------------------------------
// ncu_test — single launch for correctness verification
// ---------------------------------------------------------------------------

template <class dtypeA, class dtypeB, class dtypeC, class dtypeMask,
          int Tile_M, int Tile_N, int Tile_K, int Mma_M, int Mma_N, int Mma_K>
void ncu_test(int threadblock_num_x, int threadblock_num_y, int thread_num,
              int M, int N, int K, int Mo, int nnz_aligned, int nnz_block,
              dtypeB* dB_val, dtypeC* dC_val, int* dval_sidx, int* dval_soff,
              dtypeMask* dval_mco_mask, int* dval_mco_off, dtypeA* dval_mco_val) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<Tile_M>{}, Int<Tile_N>{}, Int<Tile_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(Int<Mma_M>{}, Int<Mma_N>{}, Int<Mma_K>{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<Tile_M / Mma_M>{}, Int<Tile_N / Mma_N>{}, Int<Tile_K / Mma_K>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    using nmask_per_tile = Int<Tile_M * Tile_K / 64>;
    using tile_mco_val_size = Int<Tile_M * Tile_K>;

    // ---- Profiling: collect from multiple blocks ----
    int total_blocks = threadblock_num_x * threadblock_num_y;
    int prof_nblocks = min(total_blocks, 64);  // default: up to 64 blocks
    // Override via env var PROF_NBLOCKS
    if (const char* env = getenv("PROF_NBLOCKS")) {
        prof_nblocks = min(atoi(env), total_blocks);
        if (prof_nblocks <= 0) prof_nblocks = total_blocks;
    }
    fprintf(stderr, "Profiling %d / %d blocks\n", prof_nblocks, total_blocks);

    ProfileEntry* d_profile;
    int* d_profile_count;
    CHECK_CUDA2(cudaMalloc(&d_profile, sizeof(ProfileEntry) * MAX_PROFILE_ENTRIES * prof_nblocks));
    CHECK_CUDA2(cudaMalloc(&d_profile_count, sizeof(int) * prof_nblocks));

    static const char* op_names[] = {
        "G2rSparseIndexLoadOp", "G2rSparseMcoOffLoadOp", "G2rSparseMcoMaskLoadOp",
        "G2sSparseMcoValLoadOp", "G2sMatrixBLoadOp", "S2sRestoreMatrixAOp",
        "S2rAValLoadOp", "S2rBValLoadOp", "CalculateOp"
    };

    auto run_and_print = [&](int mode, const char* label) {
        CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(dtypeC) * Mo * Tile_M * N));
        CHECK_CUDA2(cudaMemset(d_profile_count, 0, sizeof(int) * prof_nblocks));

        bitbsr_spmm_profiling_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                   nmask_per_tile, tile_mco_val_size>
            <<<grid, block>>>(dB_val, dC_val, dval_mco_mask, dval_mco_off, dval_mco_val,
                              dval_sidx, dval_soff, K, M, Mo, N, nnz_aligned, nnz_block,
                              d_profile, d_profile_count, mode, prof_nblocks);
        CHECK_CUDA2(cudaDeviceSynchronize());
        CHECK_CUDA2(cudaGetLastError());

        // Read back per-block counts
        int* h_counts = new int[prof_nblocks];
        CHECK_CUDA2(cudaMemcpy(h_counts, d_profile_count, sizeof(int) * prof_nblocks,
                               cudaMemcpyDeviceToHost));

        // Read back all entries
        ProfileEntry* h_profile = new ProfileEntry[MAX_PROFILE_ENTRIES * prof_nblocks];
        CHECK_CUDA2(cudaMemcpy(h_profile, d_profile,
                               sizeof(ProfileEntry) * MAX_PROFILE_ENTRIES * prof_nblocks,
                               cudaMemcpyDeviceToHost));

        // Print per-block entries (skip empty blocks)
        int active = 0;
        for (int b = 0; b < prof_nblocks; b++) {
            int n = min(h_counts[b], MAX_PROFILE_ENTRIES);
            if (n == 0) continue;  // block had k=0, no work
            active++;
            ProfileEntry* base = h_profile + b * MAX_PROFILE_ENTRIES;
            int iter = 0;
            for (int i = 0; i < n; i++) {
                if (i > 0 && base[i].op_id <= base[i-1].op_id) iter++;
                printf("%s %d %d %d %s %u\n", label, b, iter, base[i].op_id,
                       op_names[base[i].op_id], base[i].t_issue);
            }
        }

        fprintf(stderr, "%s: %d/%d blocks active\n", label, active, prof_nblocks);
        delete[] h_counts;
        delete[] h_profile;
    };

    // ---- Mode 0: T_issue (produces correct C output) ----
    run_and_print(0, "PROFILE");

    // Save correct C before sandwich mode corrupts it
    // (sandwich runs CalculateOp multiple times, accumulating wrong values)
    {
        size_t C_bytes = sizeof(dtypeC) * threadblock_num_x * Tile_M * N;
        dtypeC* dC_saved;
        CHECK_CUDA2(cudaMalloc(&dC_saved, C_bytes));
        CHECK_CUDA2(cudaMemcpy(dC_saved, dC_val, C_bytes, cudaMemcpyDeviceToDevice));

        // ---- Mode 1: T_tail sandwich (corrupts C) ----
        run_and_print(1, "SANDWICH");

        // Restore correct C
        CHECK_CUDA2(cudaMemcpy(dC_val, dC_saved, C_bytes, cudaMemcpyDeviceToDevice));
        cudaFree(dC_saved);
    }

    cudaFree(d_profile);
    cudaFree(d_profile_count);
}

// ---------------------------------------------------------------------------
// repeat_test — timing (runs profiling kernel, reports wall time)
// ---------------------------------------------------------------------------

template <class dtypeA, class dtypeB, class dtypeC, class dtypeMask,
          int Tile_M, int Tile_N, int Tile_K, int Mma_M, int Mma_N, int Mma_K>
void repeat_test(int threadblock_num_x, int threadblock_num_y, int thread_num,
                 int M, int N, int K, int Mo, int nnz_aligned, int nnz_block,
                 dtypeB* dB_val, dtypeC* dC_val, int* dval_sidx, int* dval_soff,
                 dtypeMask* dval_mco_mask, int* dval_mco_off, dtypeA* dval_mco_val,
                 int warmup_time, int execute_time) {
    dim3 grid(threadblock_num_x, threadblock_num_y);
    dim3 block(thread_num);

    constexpr auto blk_mnk = make_shape(Int<Tile_M>{}, Int<Tile_N>{}, Int<Tile_K>{});
    using BLK_MNK = decltype(blk_mnk);
    constexpr auto mma_mnk = make_shape(Int<Mma_M>{}, Int<Mma_N>{}, Int<Mma_K>{});
    using MMA_MNK = decltype(mma_mnk);
    constexpr auto blk_mma_mnk =
      make_shape(Int<Tile_M / Mma_M>{}, Int<Tile_N / Mma_N>{}, Int<Tile_K / Mma_K>{});
    using BLK_MMA_MNK = decltype(blk_mma_mnk);
    constexpr auto warp_mnk = make_shape(_1{}, _1{}, _1{});
    using WARP_MNK = decltype(warp_mnk);

    using nmask_per_tile = Int<Tile_M * Tile_K / 64>;
    using tile_mco_val_size = Int<Tile_M * Tile_K>;

    struct timeval t1, t2;
    {
        for (int i = 0; i < warmup_time; i++) {
            bitbsr_spmm_profiling_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                       nmask_per_tile, tile_mco_val_size>
              <<<grid, block>>>(dB_val, dC_val, dval_mco_mask, dval_mco_off, dval_mco_val,
                                dval_sidx, dval_soff, K, M, Mo, N, nnz_aligned, nnz_block,
                                nullptr, nullptr, 0, 0);
        }
        CHECK_CUDA2(cudaDeviceSynchronize());
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; i++) {
            bitbsr_spmm_profiling_kernel<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                       nmask_per_tile, tile_mco_val_size>
              <<<grid, block>>>(dB_val, dC_val, dval_mco_mask, dval_mco_off, dval_mco_val,
                                dval_sidx, dval_soff, K, M, Mo, N, nnz_aligned, nnz_block,
                                nullptr, nullptr, 0, 0);
        }
        CHECK_CUDA2(cudaDeviceSynchronize());
        gettimeofday(&t2, NULL);
        double mykernel_time =
          ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time;
        printf("mykernel_time: %f ms\n", mykernel_time);
    }
}

// ---------------------------------------------------------------------------
// bitbsr_spmm_test — same interface as host_program.cu
// ---------------------------------------------------------------------------

template <class dtypeA, class dtypeB, class dtypeC, class dtypeMask,
          int Tile_M, int Tile_N, int Tile_K, int Mma_M, int Mma_N, int Mma_K>
void bitbsr_spmm_test(int M, int N, int K,
                   dtypeB* B_val, dtypeC* C_val,
                   int* val_sidx, int* val_soff, int* val_mco_off,
                   dtypeMask* val_mco_mask, dtypeA* val_mco_val,
                   bool is_ncu_test, int warmup_time, int execute_time) {

    dtypeB* dB_val;
    dtypeC* dC_val;
    int* dval_sidx;
    int* dval_soff;
    int* dval_mco_off;
    dtypeMask* dval_mco_mask;
    dtypeA* dval_mco_val;

    assert(N % Tile_N == 0);

    int Mo = DIVUP(M, Tile_M);
    int M_ALIGNED = Mo * Tile_M;
    int nnz_block = val_soff[Mo];
    int nnz_aligned = ALIGN(val_mco_off[nnz_block], 8);
    constexpr int nmask_per_tile = Tile_M * Tile_K / 64;

    CHECK_CUDA2(cudaMalloc(&dB_val, sizeof(dtypeB) * K * N));
    CHECK_CUDA2(cudaMalloc(&dC_val, sizeof(dtypeC) * M_ALIGNED * N));
    CHECK_CUDA2(cudaMalloc(&dval_soff, sizeof(int) * (Mo + 1)));
    CHECK_CUDA2(cudaMalloc(&dval_sidx, sizeof(int) * nnz_block));
    CHECK_CUDA2(cudaMalloc(&dval_mco_mask, sizeof(dtypeMask) * nmask_per_tile * nnz_block));
    CHECK_CUDA2(cudaMalloc(&dval_mco_off, sizeof(int) * (nnz_block + 1)));
    CHECK_CUDA2(cudaMalloc(&dval_mco_val, sizeof(dtypeA) * nnz_aligned));

    CHECK_CUDA2(cudaMemcpy(dB_val, B_val, sizeof(dtypeB) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(dtypeC) * M_ALIGNED * N));
    CHECK_CUDA2(cudaMemcpy(dval_soff, val_soff, sizeof(int) * (Mo + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_sidx, val_sidx, sizeof(int) * nnz_block, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_mco_mask, val_mco_mask,
                           sizeof(dtypeMask) * nmask_per_tile * nnz_block, cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_mco_off, val_mco_off, sizeof(int) * (nnz_block + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(cudaMemcpy(dval_mco_val, val_mco_val, sizeof(dtypeA) * nnz_aligned, cudaMemcpyHostToDevice));

    int threadblock_num_x = Mo;
    int threadblock_num_y = N / Tile_N;
    int thread_num = 32;

    if (!is_ncu_test) {
        repeat_test<dtypeA, dtypeB, dtypeC, dtypeMask, Tile_M, Tile_N, Tile_K, Mma_M, Mma_N, Mma_K>(
          threadblock_num_x, threadblock_num_y, thread_num, M, N, K, Mo, nnz_aligned, nnz_block,
          dB_val, dC_val, dval_sidx, dval_soff, dval_mco_mask, dval_mco_off, dval_mco_val,
          warmup_time, execute_time);
    }

    // Reset C and run once more for profiling + correctness
    CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(dtypeC) * M_ALIGNED * N));
    ncu_test<dtypeA, dtypeB, dtypeC, dtypeMask, Tile_M, Tile_N, Tile_K, Mma_M, Mma_N, Mma_K>(
      threadblock_num_x, threadblock_num_y, thread_num, M, N, K, Mo, nnz_aligned, nnz_block,
      dB_val, dC_val, dval_sidx, dval_soff, dval_mco_mask, dval_mco_off, dval_mco_val);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
    CHECK_CUDA2(cudaMemcpy(C_val, dC_val, sizeof(dtypeC) * M * N, cudaMemcpyDeviceToHost));
}

// Explicit template instantiation (must match main.cu's extern declaration)
template void
bitbsr_spmm_test<float, float, float, uint64_t, 16, 64, 8, 16, 8, 8>(int M, int N, int K,
    float* B_val, float* C_val, int* val_sidx, int* val_soff, int* val_mco_off,
    uint64_t* val_mco_mask, float* val_mco_val, bool is_ncu_test, int warmup_time, int execute_time);
