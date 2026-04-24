/**
 * instrumentation.cuh — Minimal clock() instrumentation for pipelined kernels
 * =============================================================================
 *
 * Design: static __shared__ buffer + register counter, passed through the
 * function call chain.  All threads participate (no threadIdx check — single
 * warp kernel).  Flush to GMEM only for block (0,0).
 *
 * Per-op overhead: 2 clock reads (CS2R) + 1 inline PTX st.shared.v2 + 1 IADD.
 * No dynamic shared memory, no divergence, no bounds check.
 *
 * NOTE: Uses inline PTX for the shared memory store to work around a ptxas
 * codegen issue where compiler-generated STS instructions between cp.async
 * operations can corrupt address calculations on certain pipeline schedules.
 */

#pragma once

#include <stdint.h>

// ---------------------------------------------------------------------------
// Trace entry — 8 bytes: {t_start, t_end}
// ---------------------------------------------------------------------------

struct InstrTrace {
    uint32_t t_start;
    uint32_t t_end;
};

// Max trace entries per block.  512 * 8 = 4 KB static SMEM.
#define MAX_INSTR_ENTRIES 512

// ---------------------------------------------------------------------------
// Clock read — inline asm volatile to prevent compiler reordering
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t instr_clock() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(r));
    return r;
}

// ---------------------------------------------------------------------------
// INSTR_BEGIN / INSTR_END
//
// Usage:  { INSTR_BEGIN(); op_call(); INSTR_END(_idx, _buf); }
//
// _idx  : int& — register counter (incremented after each call)
// _buf  : InstrTrace* — pointer to static __shared__ buffer
// ---------------------------------------------------------------------------

#define INSTR_BEGIN() uint32_t _instr_t1 = instr_clock()

#define INSTR_END(_idx, _buf) \
    do { \
        uint32_t _instr_t2 = instr_clock(); \
        uint32_t _smem_addr; \
        asm("{ .reg .u64 _p; cvta.to.shared.u64 _p, %1; cvt.u32.u64 %0, _p; }" \
            : "=r"(_smem_addr) : "l"((size_t)&(_buf)[_idx])); \
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};" \
                     : : "r"(_smem_addr), "r"(_instr_t1), "r"(_instr_t2) \
                     : "memory"); \
        ++(_idx); \
    } while(0)
