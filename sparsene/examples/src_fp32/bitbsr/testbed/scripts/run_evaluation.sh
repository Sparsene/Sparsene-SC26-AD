#!/bin/bash
# ===========================================================================
# run_evaluation.sh — End-to-end simulator evaluation pipeline
# ===========================================================================
#
# Given a result.json (plan IDs + kernel times from a colleague) and the
# plan_searcher.py that generated those plans, this script:
#   1. Regenerates the manifest (ID -> stages+shifts mapping)
#   2. Converts result.json to simulate_plans.py input format
#   3. Runs profiling kernel to measure per-op T_issue/T_tail
#   4. Runs simulation on all plans with profiled values
#   5. Reports accuracy metrics (Kendall Tau, Spearman, top-k oracle)
#
# Prerequisites:
#   - CUDA toolkit (nvcc) in PATH
#   - sparsene Python package accessible (set SPARSENE_PYTHON below)
#   - graphviz Python package: pip install graphviz
#   - This testbed with host_program_profiling.cu, kernel.inc, etc.
#
# Usage:
#   cd sparsene/examples/src_fp32/bitbsr/testbed
#   bash scripts/run_evaluation.sh /path/to/result.json [options]
#
# Options:
#   --plan-searcher PATH   Path to plan_searcher.py (default: this testbed's)
#   --min-stages N         Min pipeline stages (default: 2)
#   --max-stages N         Max pipeline stages (default: 3)
#   --min-ops N            Min ops per stage (default: 2)
#   --max-ops N            Max ops per stage (default: 4)
#   --max-shift N          Max shift value (default: 3)
#   --arch ARCH            CUDA arch (default: sm_89)
#   --skip-profile         Skip profiling, use placeholder values
#   --profiles PATH        Use existing op_profiles.json instead of profiling
#   -M N -K N              Matrix dimensions (default: 1024 1024)
#   --sparsity F           Sparsity (default: 0.9)
#   -o DIR                 Output directory (default: eval_results)
# ===========================================================================

set -e

# ---- Defaults ----
TESTBED="$(cd "$(dirname "$0")/.."; pwd)"
SCRIPTS="$TESTBED/scripts"
SPARSENE_PYTHON="${SPARSENE_PYTHON:-$(cd "$TESTBED/../../../../python" 2>/dev/null && pwd || echo "")}"
PLAN_SEARCHER="$SCRIPTS/plan_searcher.py"
MIN_STAGES=2; MAX_STAGES=3; MIN_OPS=2; MAX_OPS=4; MAX_SHIFT=3
ARCH="sm_89"
SKIP_PROFILE=0
PROFILES_PATH=""
M=1024; K=1024; N=64; SPARSITY=0.9
OUTPUT_DIR="eval_results"
RESULT_JSON=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan-searcher) PLAN_SEARCHER="$2"; shift 2;;
        --min-stages) MIN_STAGES="$2"; shift 2;;
        --max-stages) MAX_STAGES="$2"; shift 2;;
        --min-ops) MIN_OPS="$2"; shift 2;;
        --max-ops) MAX_OPS="$2"; shift 2;;
        --max-shift) MAX_SHIFT="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --skip-profile) SKIP_PROFILE=1; shift;;
        --profiles) PROFILES_PATH="$2"; shift 2;;
        -M) M="$2"; shift 2;;
        -K) K="$2"; shift 2;;
        --sparsity) SPARSITY="$2"; shift 2;;
        -o) OUTPUT_DIR="$2"; shift 2;;
        -*) echo "Unknown option: $1"; exit 1;;
        *) RESULT_JSON="$1"; shift;;
    esac
done

if [ -z "$RESULT_JSON" ]; then
    echo "Usage: bash scripts/run_evaluation.sh /path/to/result.json [options]"
    echo ""
    echo "Run 'bash scripts/run_evaluation.sh --help' or read the script header for options."
    exit 1
fi

if [ ! -f "$RESULT_JSON" ]; then
    echo "ERROR: result.json not found: $RESULT_JSON"
    exit 1
fi

if [ -n "$SPARSENE_PYTHON" ]; then
    export PYTHONPATH="$SPARSENE_PYTHON:$PYTHONPATH"
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Simulator Evaluation Pipeline"
echo "============================================================"
echo " Result JSON:     $RESULT_JSON"
echo " Plan searcher:   $PLAN_SEARCHER"
echo " Enum params:     stages=$MIN_STAGES-$MAX_STAGES, ops=$MIN_OPS-$MAX_OPS, shift<=$MAX_SHIFT"
echo " CUDA arch:       $ARCH"
echo " Matrix:          M=$M K=$K N=$N sparsity=$SPARSITY"
echo " Output:          $OUTPUT_DIR/"
echo " Skip profiling:  $SKIP_PROFILE"
echo "============================================================"
echo ""

# ===========================================================================
# Step 1: Generate manifest + convert result.json
# ===========================================================================
echo ">>> Step 1: Generate manifest and convert result.json"

PLANS_JSON="$OUTPUT_DIR/plans_for_simulation.json"

python3 "$SCRIPTS/generate_manifest.py" \
    --plan-searcher "$PLAN_SEARCHER" \
    --min-stages "$MIN_STAGES" \
    --max-stages "$MAX_STAGES" \
    --min-ops-per-stage "$MIN_OPS" \
    --max-ops-per-stage "$MAX_OPS" \
    --max-shift "$MAX_SHIFT" \
    --result-json "$RESULT_JSON" \
    -o "$PLANS_JSON"

echo ""

# ===========================================================================
# Step 2: Profiling (optional)
# ===========================================================================
OP_PROFILES=""

if [ -n "$PROFILES_PATH" ]; then
    echo ">>> Step 2: Using provided profiles: $PROFILES_PATH"
    OP_PROFILES="$PROFILES_PATH"
elif [ "$SKIP_PROFILE" -eq 1 ]; then
    echo ">>> Step 2: Skipping profiling (using placeholder values)"
else
    echo ">>> Step 2: Running profiling kernel"

    NVCC_FLAGS="-std=c++17 -O3 -m64 -g -lineinfo -Xcompiler -fopenmp-simd --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 -Df32 -arch=$ARCH"
    INC="-I$TESTBED -I$TESTBED/src -I$TESTBED/cutlass/include -I$TESTBED/cutlass/examples/common -I$TESTBED/cutlass/tools/util/include"
    LINK="-lcusparse -lcublas"

    BUILD="$OUTPUT_DIR/build"
    mkdir -p "$BUILD"

    echo "  Compiling..."
    nvcc $NVCC_FLAGS $INC -dc -o "$BUILD/utils.o" "$TESTBED/src/utils.cu" 2>&1 | grep -v "^$"
    nvcc $NVCC_FLAGS $INC -dc -o "$BUILD/mmio.o" "$TESTBED/src/mmio.cu" 2>&1 | grep -v "^$"
    nvcc $NVCC_FLAGS $INC -dc -o "$BUILD/mmio_highlevel.o" "$TESTBED/src/mmio_highlevel.cu" 2>&1 | grep -v "^$"
    nvcc $NVCC_FLAGS $INC -dc -o "$BUILD/main.o" "$TESTBED/main.cu" 2>&1 | grep -v "^$"
    nvcc $NVCC_FLAGS $INC -dc -o "$BUILD/host_program_profiling.o" "$TESTBED/host_program_profiling.cu" 2>&1 | grep -v "^$"
    nvcc $NVCC_FLAGS -o "$BUILD/bitbsr_profiling" \
        "$BUILD/host_program_profiling.o" "$BUILD/main.o" \
        "$BUILD/utils.o" "$BUILD/mmio.o" "$BUILD/mmio_highlevel.o" \
        $LINK 2>&1 | grep -v "^$"
    echo "  Build OK"

    echo "  Running profiling kernel (64 blocks)..."
    PROF_NBLOCKS=64 "$BUILD/bitbsr_profiling" \
        -M "$M" -K "$K" -N "$N" -sparsity "$SPARSITY" -mtx_flag 0 -ncu 1 \
        > "$OUTPUT_DIR/profiling_raw.txt" 2>&1

    echo "  Parsing profiling output..."
    python3 -c "
import sys, statistics as stat, json
from collections import defaultdict
sys.path.insert(0, '$SCRIPTS')
from perf_model import make_bitbsr_spmm_profiles, save_profiles, BITBSR_SPMM_DEPENDENCIES

raw = open('$OUTPUT_DIR/profiling_raw.txt').read()
ops = ['G2rSparseIndexLoadOp', 'G2rSparseMcoOffLoadOp', 'G2rSparseMcoMaskLoadOp',
       'G2sSparseMcoValLoadOp', 'G2sMatrixBLoadOp', 'S2sRestoreMatrixAOp',
       'S2rAValLoadOp', 'S2rBValLoadOp', 'CalculateOp']

prof = defaultdict(lambda: defaultdict(list))
sand = defaultdict(lambda: defaultdict(list))
for line in raw.splitlines():
    p = line.split()
    if len(p) >= 6:
        if p[0] == 'PROFILE' and int(p[2]) > 0:
            prof[p[4]][int(p[1])].append(int(p[5]))
        elif p[0] == 'SANDWICH' and int(p[2]) > 0:
            sand[int(p[3])][int(p[1])].append(int(p[5]))

t_issue = {}
for op in ops:
    meds = [stat.median(v) for v in prof[op].values() if v]
    t_issue[op] = stat.mean(meds) if meds else 0

pairs = [(0,4),(1,3),(2,5),(3,5),(4,7),(5,6),(6,8),(7,8),(8,-1)]
t_tail = {}
for pi, ci in pairs:
    pn, cn = ops[pi], (ops[ci] if ci >= 0 else None)
    tails = []
    for blk in sand[pi]:
        v = sand[pi][blk]
        if not v: continue
        sw = stat.median(v)
        ta = stat.median(prof[pn].get(blk, [0]))
        tb = stat.median(prof[cn].get(blk, [0])) if cn else 0
        tails.append(max(sw - ta - tb, 0))
    t_tail[pn] = stat.mean(tails) if tails else 0

profiles = make_bitbsr_spmm_profiles()
for op in ops:
    if op in t_issue: profiles[op].T_issue = t_issue[op]
    if op in t_tail: profiles[op].T_tail = t_tail[op]

out_path = '$OUTPUT_DIR/op_profiles.json'
save_profiles(profiles, out_path, dependencies=BITBSR_SPMM_DEPENDENCIES,
              metadata={'source': 'profiling kernel, 64-block average', 'M': $M, 'K': $K, 'N': $N, 'sparsity': $SPARSITY})

print()
print('  Per-op profiled values:')
print(f'  {\"Op\":<30s} {\"T_issue\":>8s} {\"T_tail\":>8s}')
print(f'  {\"-\"*30} {\"-\"*8} {\"-\"*8}')
for op in ops:
    print(f'  {op:<30s} {t_issue[op]:8.0f} {t_tail[op]:8.0f}')
print(f'  Saved to {out_path}')
"

    OP_PROFILES="$OUTPUT_DIR/op_profiles.json"
fi

echo ""

# ===========================================================================
# Step 3: Run simulation
# ===========================================================================
echo ">>> Step 3: Running simulation on all plans"

SIM_ARGS="$PLANS_JSON"
if [ -n "$OP_PROFILES" ]; then
    SIM_ARGS="$SIM_ARGS --profiles $OP_PROFILES"
fi
SIM_ARGS="$SIM_ARGS -o $OUTPUT_DIR/simulation_results.json"

python3 "$SCRIPTS/simulate_plans.py" $SIM_ARGS

echo ""
echo "============================================================"
echo " Results saved to:"
echo "   $OUTPUT_DIR/plans_for_simulation.json  (input with stages+shifts)"
if [ -n "$OP_PROFILES" ]; then
    echo "   $OP_PROFILES  (profiled op timings)"
fi
echo "   $OUTPUT_DIR/simulation_results.json   (per-plan predictions + metrics)"
echo "============================================================"
