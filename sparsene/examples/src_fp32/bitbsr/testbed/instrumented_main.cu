/**
 * instrumented_main.cu — main.cu with clock() instrumentation for pipeline tracing
 *
 * Diff from main.cu:
 *   1. Replaced extern bitbsr_spmm_test with inline definition (from host_program.cu)
 *      that adds trace buffer params to the instrumented kernel launch
 *   2. Added cutlass includes + kernel_instrumented.inc
 *   Everything else (convertToBitBSR, main, cusparse, verify) is verbatim from main.cu.
 *
 * Build: cmake target bitbsr-instrumented (links with src/*.cu for utils, mmio, etc.)
 */

#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "common.h"
#include "utils.h"

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "kernel_instrumented.inc"

using namespace cute;
using namespace std;

#define OFFSET_ROW(row, col, lda) ((row) * (lda) + (col))
#define OFFSET_COL(row, col, lda) ((col) * (lda) + (row))

#define DIVUP(a, b) (((a) - 1) / (b) + 1)
#define ALIGN(a, b) (DIVUP(a, b) * (b))

// ---------------------------------------------------------------------------
// Instrumented bitbsr_spmm_test (replaces extern template from host_program.cu)
// Same device alloc + kernel launch, but with trace buffer handling.
// ---------------------------------------------------------------------------

template <class dtypeA,
          class dtypeB,
          class dtypeC,
          class dtypeMask,
          int Tile_M,
          int Tile_N,
          int Tile_K,
          int Mma_M,
          int Mma_N,
          int Mma_K>
void bitbsr_spmm_test(int M,
                   int N,
                   int K,
                   dtypeB* B_val,
                   dtypeC* C_val,
                   int* val_sidx,
                   int* val_soff,
                   int* val_mco_off,
                   dtypeMask* val_mco_mask,
                   dtypeA* val_mco_val,
                   bool is_ncu_test,
                   int warmup_time,
                   int execute_time) {

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
    CHECK_CUDA2(
      cudaMemcpy(dval_mco_off, val_mco_off, sizeof(int) * (nnz_block + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA2(
      cudaMemcpy(dval_mco_val, val_mco_val, sizeof(dtypeA) * nnz_aligned, cudaMemcpyHostToDevice));

    int threadblock_num_x = Mo;
    int threadblock_num_y = N / Tile_N;
    int thread_num = 32;

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

    using nmask_per_tile_t = Int<Tile_M * Tile_N / 64>;
    using tile_mco_val_size = Int<Tile_M * Tile_N>;

    // ---- Multi-block trace collection ----
    int total_blocks = grid.x * grid.y;
    int instr_nblocks = min(total_blocks, 64);  // default: up to 64
    if (const char* env = getenv("INSTR_NBLOCKS")) {
        instr_nblocks = atoi(env);
        if (instr_nblocks <= 0) instr_nblocks = total_blocks;
        instr_nblocks = min(instr_nblocks, total_blocks);
    }
    fprintf(stderr, "Tracing %d / %d blocks\n", instr_nblocks, total_blocks);

    InstrTrace* d_trace;
    int* d_count;
    CHECK_CUDA2(cudaMalloc(&d_trace, sizeof(InstrTrace) * MAX_INSTR_ENTRIES * instr_nblocks));
    CHECK_CUDA2(cudaMalloc(&d_count, sizeof(int) * instr_nblocks));

    // ---- Warmup (nblocks=0 -> flush is skipped) ----
    for (int w = 0; w < warmup_time; w++) {
        CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(dtypeC) * M_ALIGNED * N));
        bitbsr_spmm_kernel_tf32tf32fp32<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                      nmask_per_tile_t, tile_mco_val_size>
            <<<grid, block>>>(dB_val, dC_val, dval_mco_mask, dval_mco_off, dval_mco_val,
                              dval_sidx, dval_soff, K, M, Mo, N, nnz_aligned, nnz_block,
                              nullptr, nullptr, 0);
        CHECK_CUDA2(cudaDeviceSynchronize());
    }

    // ---- Measurement run ----
    CHECK_CUDA2(cudaMemset(dC_val, 0, sizeof(dtypeC) * M_ALIGNED * N));
    CHECK_CUDA2(cudaMemset(d_count, 0, sizeof(int) * instr_nblocks));

    fprintf(stderr, "Launching instrumented kernel (%d x %d blocks, %d threads)...\n",
            grid.x, grid.y, thread_num);

    bitbsr_spmm_kernel_tf32tf32fp32<BLK_MNK, MMA_MNK, BLK_MMA_MNK, WARP_MNK,
                                  nmask_per_tile_t, tile_mco_val_size>
        <<<grid, block>>>(dB_val, dC_val, dval_mco_mask, dval_mco_off, dval_mco_val,
                          dval_sidx, dval_soff, K, M, Mo, N, nnz_aligned, nnz_block,
                          d_trace, d_count, instr_nblocks);

    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());

    // ---- Read back C (for verification in main) ----
    CHECK_CUDA2(cudaMemcpy(C_val, dC_val, sizeof(dtypeC) * M * N, cudaMemcpyDeviceToHost));

    // ---- Read back and print per-block traces ----
    int* h_counts = new int[instr_nblocks];
    CHECK_CUDA2(cudaMemcpy(h_counts, d_count, sizeof(int) * instr_nblocks,
                           cudaMemcpyDeviceToHost));

    InstrTrace* h_trace = new InstrTrace[MAX_INSTR_ENTRIES * instr_nblocks];
    CHECK_CUDA2(cudaMemcpy(h_trace, d_trace,
                           sizeof(InstrTrace) * MAX_INSTR_ENTRIES * instr_nblocks,
                           cudaMemcpyDeviceToHost));

    int total_entries = 0;
    int active_blocks = 0;
    for (int b = 0; b < instr_nblocks; b++) {
        int n = min(h_counts[b], MAX_INSTR_ENTRIES);
        total_entries += n;
        if (n > 0) active_blocks++;
    }

    printf("# Instrumented Pipeline Trace (%d entries, %d/%d blocks active)\n",
           total_entries, active_blocks, instr_nblocks);
    printf("# RAWTRACE <block> <idx> <t_start> <t_end>\n");
    for (int b = 0; b < instr_nblocks; b++) {
        int n = min(h_counts[b], MAX_INSTR_ENTRIES);
        if (n == 0) continue;  // skip empty blocks (k=0)
        InstrTrace* base = h_trace + b * MAX_INSTR_ENTRIES;
        for (int i = 0; i < n; i++) {
            printf("RAWTRACE %d %d %u %u\n", b, i, base[i].t_start, base[i].t_end);
        }
    }

    fprintf(stderr, "Collected %d entries across %d/%d blocks.\n",
            total_entries, active_blocks, instr_nblocks);

    // ---- Cleanup ----
    delete[] h_counts;
    delete[] h_trace;
    cudaFree(d_trace);
    cudaFree(d_count);
    cudaFree(dB_val);
    cudaFree(dC_val);
    cudaFree(dval_sidx);
    cudaFree(dval_soff);
    cudaFree(dval_mco_mask);
    cudaFree(dval_mco_off);
    cudaFree(dval_mco_val);
}

// ---------------------------------------------------------------------------
// Everything below is verbatim from main.cu
// ---------------------------------------------------------------------------

int binarySearch(int* arr, int l, int r, int target) {
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target)
            return mid;
        if (arr[mid] < target)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return -1;
}

void initVector(float* vec, int length, float sparsity) {
    if (sparsity < 0.0f || sparsity > 1.0f)
        return;
    auto seed = std::time(nullptr);
    std::srand(1234);
    cout << seed << endl;
    for (int i = 0; i < length; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < sparsity) {
            vec[i] = float(0.0f);
        } else {
            vec[i] = float(rand() % 5 + 1);
        }
    }
}

vector<uint64_t> convertToUInt64Vector(const vector<bool>& bitset) {
    vector<uint64_t> uint64Vector;
    uint64_t current = 0;
    int bitIndex = 0;

    for (size_t i = 0; i < bitset.size(); ++i) {
        current |= (static_cast<uint64_t>(bitset[i]) << bitIndex);
        ++bitIndex;
        if (bitIndex == 64) {
            uint64Vector.push_back(current);
            current = 0;
            bitIndex = 0;
        }
    }
    if (bitIndex > 0) {
        uint64Vector.push_back(current);
    }
    return uint64Vector;
}

void convertToBitBSR(int M,
                     int N,
                     int K,
                     int nnz,
                     int* csrPtrA,
                     int* csrColIdxA,
                     float* csrValA,
                     float* B,
                     float* C,
                     bool is_ncu_test,
                     int warmup_time,
                     int execute_time) {
    using dtypeA = float;
    using dtypeB = float;
    using dtypeC = float;
    using dtypeMask = uint64_t;

    // 1. Create ptrs
    dtypeB* B_val;
    dtypeC* C_val;
    int* val_sidx;
    int* val_soff;
    int* val_mco_off;
    dtypeMask* val_mco_mask;
    dtypeA* val_mco_val;

    B_val = B;
    C_val = C;

    const int Blk_M = 16;
    const int Blk_K = 8;

    // 2. Convert

    auto swizzler =
      cute::composition(cute::Swizzle<1, 3, 3>{}, cute::make_layout(cute::make_shape(Blk_M, Blk_K),
                                                                    cute::make_stride(Blk_K, 1)));
    int num_row_windows = DIVUP(M, Blk_M);

    vector<vector<int>> nz_blk_indices(num_row_windows);
    vector<vector<uint64_t>> nz_blk_mask;
    vector<vector<dtypeA>> nz_blk_val;

    for (int rw_idx = 0; rw_idx < num_row_windows; rw_idx++) {
        int row_start = rw_idx * Blk_M;
        int row_end = min(row_start + Blk_M, M);

        map<int, map<int, dtypeA>> nz_blk_idx_val_in_window;
        for (int row = row_start; row < row_end; row++) {
            for (int j = csrPtrA[row]; j < csrPtrA[row + 1]; j++) {
                int col = csrColIdxA[j];
                int blk_idx = col / Blk_K;
                int idx = swizzler(cute::make_coord(row % Blk_M, col % Blk_K));
                dtypeA val = csrValA[j];
                nz_blk_idx_val_in_window[blk_idx][idx] = val;
            }
        }

        for (auto& [blk_idx, idx_vals] : nz_blk_idx_val_in_window) {
            nz_blk_indices[rw_idx].push_back(blk_idx);
            vector<bool> bitmask(Blk_M * Blk_K, false);
            vector<dtypeA> vals;
            for (auto& [idx, val] : idx_vals) {
                bitmask[idx] = true;
                vals.push_back(val);
            }
            nz_blk_mask.push_back(convertToUInt64Vector(bitmask));
            while (vals.size() % 4 != 0) {
                // padding with 0 to align with 4*fp32 (32 bytes)
                vals.push_back(float(0.0f));
            }
            nz_blk_val.push_back(vals);
        }
    }

    // 3. Accumulate offsets and flatten values
    vector<int> vval_soff;
    vector<int> vval_sidx;
    vector<int> vval_mco_off;
    vector<uint64_t> vval_mco_mask;
    vector<dtypeA> vval_mco_val;

    int soff_acc = 0;
    int mco_off_acc = 0;
    int blk_ptr = 0;
    for (int rw_idx = 0; rw_idx < num_row_windows; rw_idx++) {
        vval_soff.push_back(soff_acc);
        vval_sidx.insert(vval_sidx.end(), nz_blk_indices[rw_idx].begin(),
                         nz_blk_indices[rw_idx].end());
        soff_acc += nz_blk_indices[rw_idx].size();
        for (int blk_idx : nz_blk_indices[rw_idx]) {
            vval_mco_off.push_back(mco_off_acc);
            vval_mco_val.insert(vval_mco_val.end(), nz_blk_val[blk_ptr].begin(),
                                nz_blk_val[blk_ptr].end());
            mco_off_acc += nz_blk_val[blk_ptr].size();
            vval_mco_mask.insert(vval_mco_mask.end(), nz_blk_mask[blk_ptr].begin(),
                                 nz_blk_mask[blk_ptr].end());
            blk_ptr++;
        }
    }
    vval_soff.push_back(soff_acc);
    vval_mco_off.push_back(mco_off_acc);
    val_soff = vval_soff.data();
    val_sidx = vval_sidx.data();
    val_mco_off = vval_mco_off.data();
    val_mco_mask = vval_mco_mask.data();
    val_mco_val = vval_mco_val.data();

    // 4. Call the kernel
    bitbsr_spmm_test<dtypeA, dtypeB, dtypeC, dtypeMask, 16, 64, 8, 16, 8, 8>(
      M, N, K, B_val, C_val, val_sidx, val_soff, val_mco_off, val_mco_mask, val_mco_val,
      is_ncu_test, warmup_time, execute_time);
}

void verify_new(float* C_cpu, float* C_cuda, int M, int N) {
    int flag = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float cusparse_val = C_cpu[OFFSET_ROW(i, j, N)];
            float mykernel_val = C_cuda[OFFSET_ROW(i, j, N)];
            if (fabs(cusparse_val - mykernel_val) > 0.01) {
                flag++;
                if (flag < 200)
                printf("Error(%d, %d): cusp(%.1f) mykernel(%.1f)\n", i, j, cusparse_val,
                       mykernel_val);
            }
        }
    }
}

void cusparse_spmm_all(float* cu_ValA,
                       int* cu_RowPtrA,
                       int* cu_ColIdxA,
                       float* cu_B,
                       float* cu_C,
                       int M,
                       int N,
                       int K,
                       int nnz) {
    int A_num_rows = M;
    int A_num_cols = K;
    int B_num_rows = K;
    int B_num_cols = N;
    int ldb = B_num_cols; // B: row first
    int B_size = B_num_rows * ldb;
    int C_num_rows = M;
    int C_num_cols = N;
    int ldc = C_num_cols; // C: row first
    int C_size = C_num_rows * ldc;
    float alpha = 1.0f;
    float beta = 0.0f;
    // device memory management
    int* dA_rpt;
    int* dA_cid;
    float* dA_val;
    float* dB;
    float* dC;
    CHECK_CUDA(cudaMalloc(&dA_rpt, sizeof(int) * (M + 1)));
    CHECK_CUDA(cudaMalloc(&dA_cid, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc(&dA_val, sizeof(float) * nnz));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * B_size));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C_size));

    CHECK_CUDA(cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val, cu_ValA, sizeof(float) * nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, cu_B, sizeof(float) * B_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_size));

    // cuSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, nnz, dA_rpt, dA_cid, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
      matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
      matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

    CHECK_CUDA(cudaMemcpy(cu_C, dC, sizeof(float) * C_size, cudaMemcpyDeviceToHost));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_cid))
    CHECK_CUDA(cudaFree(dA_rpt))
    CHECK_CUDA(cudaFree(dA_val))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
    return;
}

int main(int argc, char** argv) {
    string filename;
    int N = 1024;
    int M = 1024;
    int K = 1024;
    float sparsity = 0.9;
    int mtx_flag = 1;
    int is_ncu_test = 0;
    int warmup_time = 100;
    int execute_time = 1000;
    parseInput(argc, argv, filename, "-filename", N, "-N", M, "-M", K, "-K", mtx_flag, "-mtx_flag",
               sparsity, "-sparsity", is_ncu_test, "-ncu", warmup_time, "-warmup", execute_time,
               "-repeat");

    cout << "Filename: " << filename << endl;
    cout << "N: " << N << endl;

    int rowA, colA;
    int nnz = 0;
    int isSymmetricA;
    int* csrPtrA;
    int* csrColIdxA;
    float* csrValA;
    float* A;
    //! input from file
    if (mtx_flag == 1) {
        mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrPtrA, &csrColIdxA, &csrValA,
                      filename.c_str());
        M = rowA;
        K = colA;
    } else { //! sparse matrix init
        A = (float*)malloc(sizeof(float) * M * K);
        initVector(A, M * K, sparsity);
        //> convert to csr
        csrPtrA = (int*)malloc(sizeof(int) * (M + 1));
        for (int i = 0; i < M; i++) {
            csrPtrA[i] = nnz;
            for (int k = 0; k < K; k++) {
                float value = (float)A[i * K + k];
                if (value != 0) {
                    nnz++;
                }
            }
        }
        csrPtrA[M] = nnz;
        csrColIdxA = (int*)malloc(sizeof(int) * nnz);
        csrValA = (float*)malloc(sizeof(float) * nnz);
        memset(csrColIdxA, 0, sizeof(int) * nnz);
        memset(csrValA, 0, sizeof(float) * nnz);
        int ptr = 0;
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float value = A[i * K + k];
                if (value != 0) {
                    csrColIdxA[ptr] = k;
                    csrValA[ptr] = value;
                    ptr++;
                }
            }
        }
    }

    printf("FLOAT: Sparse A[%d,%d] x B[%d,%d] = C[%d,%d]\n", M, K, K, N, M, N);

    float* B = (float*)malloc(sizeof(float) * K * N);
    initVec(csrValA, nnz);
    initVector(B, K * N, 0);

    float* C_cuda = (float*)malloc(sizeof(float) * M * N);
    convertToBitBSR(M, N, K, nnz, csrPtrA, csrColIdxA, csrValA, B, C_cuda, (bool)is_ncu_test,
                    warmup_time, execute_time);

    // host check
    float* C_cpu = (float*)malloc(sizeof(float) * M * N);
    memset(C_cpu, 0, sizeof(float) * M * N);
    cusparse_spmm_all(csrValA, csrPtrA, csrColIdxA, B, C_cpu, M, N, K, nnz);
    verify_new(C_cpu, C_cuda, M, N);

    printf("check nan: cusp %.1f mykernel %.1f\n", C_cpu[0], C_cuda[0]);
    return 0;
}
