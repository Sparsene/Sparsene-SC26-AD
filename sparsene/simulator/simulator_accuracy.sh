#!/bin/bash
# ===========================================================================
# simulator_accuracy.sh — internal runner for one (format, arch) combination
# ===========================================================================
#
# This is the internal runner invoked by ./run_simulator_accuracy.sh (AE T4).
# It runs the 4-step pipeline for ONE (format, arch) combination:
#   1. Generate all valid pipeline plans
#   2. Profile per-op latency (T_issue / T_tail) on the target GPU
#   3. Compile and time ALL plans on the target GPU
#   4. Simulate all plans using profiled values, compute ranking metrics
#   → Output: <output_dir>/combined_results.json
#
# Reviewers should NOT invoke this directly — use run_simulator_accuracy.sh
# instead. For CLI options, see the header of run_simulator_accuracy.sh.
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.."; pwd)"
SPARSENE_PYTHON="$REPO_ROOT/python"

FORMAT=""; ARCH=""; GPUS="0"; OUTPUT_DIR=""; GPU_NAME=""
COMPILE_JOBS=32; KEEP_BINARIES=0; MAX_PLANS=0
M=1024; K=1024; N=64; SPARSITY=0.9; WARMUP=10; REPEAT=100
SKIP_GENERATE=0; SKIP_PROFILE=0; SKIP_TIMING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --format) FORMAT="$2"; shift 2;;
        --arch) ARCH="$2"; shift 2;;
        --gpu-name) GPU_NAME="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        -o|--output) OUTPUT_DIR="$2"; shift 2;;
        --compile-jobs) COMPILE_JOBS="$2"; shift 2;;
        --max-plans) MAX_PLANS="$2"; shift 2;;
        -M) M="$2"; shift 2;;
        -K) K="$2"; shift 2;;
        -N) N="$2"; shift 2;;
        --sparsity) SPARSITY="$2"; shift 2;;
        --warmup) WARMUP="$2"; shift 2;;
        --repeat) REPEAT="$2"; shift 2;;
        --skip-generate) SKIP_GENERATE=1; shift;;
        --skip-profile) SKIP_PROFILE=1; shift;;
        --skip-timing) SKIP_TIMING=1; shift;;
        --keep-binaries) KEEP_BINARIES=1; shift;;
        -h|--help)
            echo "Usage: $0 --format <acc|bitbsr> --arch <sm_80|sm_89|...> [options]"
            echo "  --gpu-name NAME     GPU display name, e.g. 'A100', '4090', 'H100' (default: derived from --arch)"
            echo "  --gpus GPUS         GPU IDs, e.g. '0' or '4,5' (default: 0)"
            echo "  -o DIR              Output directory"
            echo "  --compile-jobs N    Parallel nvcc workers (default: 32)"
            echo "  --max-plans N       Limit generation to first N plans (for quick tests; default: all)"
            echo "  --keep-binaries     Don't delete binaries after timing"
            echo "  --skip-generate     Reuse existing plans"
            echo "  --skip-profile      Use placeholder op profiles"
            echo "  --skip-timing       Skip compile+time (profiling+sim only)"
            exit 0;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

[ -z "$FORMAT" ] && echo "ERROR: --format required (acc or bitbsr)" && exit 1
[ -z "$ARCH" ] && echo "ERROR: --arch required (e.g. sm_80, sm_89)" && exit 1

if [ -z "$GPU_NAME" ]; then
    case "$ARCH" in
        sm_80) GPU_NAME="A100";;
        sm_89) GPU_NAME="4090";;
        sm_90) GPU_NAME="H100";;
        *)     GPU_NAME="$ARCH";;
    esac
fi

case "$FORMAT" in
    acc)    TESTBED="$REPO_ROOT/examples/src_fp32/acc/testbed";;
    bitbsr) TESTBED="$REPO_ROOT/examples/src_fp32/bitbsr/testbed";;
    *)      echo "ERROR: Unknown format '$FORMAT'"; exit 1;;
esac
[ ! -d "$TESTBED" ] && echo "ERROR: $TESTBED not found" && exit 1
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$TESTBED/results_${ARCH}"

export PYTHONPATH="$SPARSENE_PYTHON:${PYTHONPATH:-}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
FIRST_GPU="${GPU_ARR[0]}"
NGPU=${#GPU_ARR[@]}

command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not in PATH"; exit 1; }
python3 -c "import graphviz" 2>/dev/null || { echo "ERROR: pip install graphviz"; exit 1; }

echo "============================================================"
echo " Simulator Evaluation: $FORMAT on $ARCH ($GPU_NAME)"
echo "============================================================"
echo " Output:  $OUTPUT_DIR"
echo " GPUs:    $GPUS ($NGPU)"
echo " Matrix:  M=$M K=$K N=$N sparsity=$SPARSITY"
echo "============================================================"

mkdir -p "$OUTPUT_DIR/obj" "$OUTPUT_DIR/outputs"
cd "$TESTBED"

NVCC_FLAGS="-std=c++17 -O3 -m64 -g -lineinfo -Xcompiler -fopenmp-simd --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 -Df32 -arch=$ARCH"
INC="-I$TESTBED -I$TESTBED/src -I$TESTBED/cutlass/include -I$TESTBED/cutlass/examples/common -I$TESTBED/cutlass/tools/util/include"
LINK="-lcusparse -lcublas"
ARGS="-M $M -K $K -N $N -sparsity $SPARSITY -mtx_flag 0 -warmup $WARMUP -repeat $REPEAT"

PLANS_DIR="$TESTBED/search_results/plans"
MANIFEST="$TESTBED/search_results/simulation/manifest.json"

# ===== Step 1: Generate plans =====
if [ "$SKIP_GENERATE" -eq 1 ] && [ -f "$MANIFEST" ]; then
    TOTAL=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
    echo ">>> Step 1: Reusing $TOTAL existing plans"
else
    echo ">>> Step 1: Generating plans..."
    FUNC=$( [ "$FORMAT" = "acc" ] && echo "acc" || echo "bitbsr" )
    python3 -c "
import sys, types, json, time
from pathlib import Path
sys.modules['graphviz'] = types.ModuleType('graphviz')
sys.path.insert(0, '$TESTBED/scripts')
from plan_searcher import $FUNC as make_kernel
from sparsene.op_gen.nvir.plan import apply_pipeline
from sparsene.op_gen.nvir.pipeline.pipeline_planner import enumerate_pipeline_plans, NeighborDependencyValidator
from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence
from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator
loop, prog = make_kernel()
graph = construct_graph_from_op_sequence(loop.body)
plans = enumerate_pipeline_plans(loop, NeighborDependencyValidator(graph),
    min_nstages=2, max_nstages=3, min_ops_per_stage=1, max_ops_per_stage=4, min_shift=1, max_shift=3)
print(f'  {len(plans)} plans enumerated')
if $MAX_PLANS > 0:
    plans = plans[:$MAX_PLANS]
    print(f'  Truncated to first {len(plans)} (via --max-plans)')
plans_dir = Path('$PLANS_DIR')
if plans_dir.exists():
    for f in plans_dir.glob('plan_*.inc'): f.unlink()
plans_dir.mkdir(parents=True, exist_ok=True)
sim_dir = Path('$TESTBED/search_results/simulation'); sim_dir.mkdir(parents=True, exist_ok=True)
manifest = []
t0 = time.time()
for i, plan in enumerate(plans):
    nl, np = make_kernel(); apply_pipeline(nl, plan)
    (plans_dir / f'plan_{i:04d}.inc').write_text(NvIrCodeGenerator().dump_nvop_program(np))
    manifest.append({'id':i,'stages':[[o.name for o in s.ops] for s in plan.stages],'shifts':list(plan.shifts)})
json.dump(manifest, open('$MANIFEST','w'), indent=2)
print(f'  Codegen: {time.time()-t0:.1f}s')
"
    TOTAL=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
fi
echo ""

# ===== Step 2: Profile =====
if [ "$SKIP_PROFILE" -eq 1 ]; then
    echo ">>> Step 2: Skipping profiling"
else
    echo ">>> Step 2: Profiling on GPU $FIRST_GPU..."
    for src in src/utils.cu src/mmio.cu src/mmio_highlevel.cu; do
        nvcc $NVCC_FLAGS $INC -dc -o "$OUTPUT_DIR/obj/$(basename ${src%.cu}.o)" "$src" 2>&1 | grep -i "^.*error" || true
    done
    nvcc $NVCC_FLAGS $INC -dc -o "$OUTPUT_DIR/obj/main.o" main.cu 2>&1 | grep -i "^.*error" || true
    COMMON="$OUTPUT_DIR/obj/utils.o $OUTPUT_DIR/obj/mmio.o $OUTPUT_DIR/obj/mmio_highlevel.o"
    nvcc $NVCC_FLAGS $INC -dc -o "$OUTPUT_DIR/obj/profiling.o" host_program_profiling.cu 2>&1 | grep -i "^.*error" || true
    nvcc $NVCC_FLAGS -o "$OUTPUT_DIR/profiling" "$OUTPUT_DIR/obj/profiling.o" "$OUTPUT_DIR/obj/main.o" $COMMON $LINK 2>&1 | grep -i "^.*error" || true

    CUDA_VISIBLE_DEVICES=$FIRST_GPU PROF_NBLOCKS=64 "$OUTPUT_DIR/profiling" \
        -M $M -K $K -N $N -sparsity $SPARSITY -mtx_flag 0 -ncu 1 > "$OUTPUT_DIR/profiling_raw.txt" 2>&1
    echo "  $(wc -l < "$OUTPUT_DIR/profiling_raw.txt") lines"
    grep "check nan" "$OUTPUT_DIR/profiling_raw.txt" || echo "  WARNING: no correctness check"

    if [ "$FORMAT" = "acc" ]; then
        MAKE_PROF="make_acc_spmm_profiles"; DEPS_VAR="ACC_SPMM_DEPENDENCIES"
        OP0="G2sSparseIndexLoadOp"
    else
        MAKE_PROF="make_bitbsr_spmm_profiles"; DEPS_VAR="BITBSR_SPMM_DEPENDENCIES"
        OP0="G2rSparseIndexLoadOp"
    fi

    python3 << PYEOF
import sys, statistics as stat, json
from collections import defaultdict
sys.path.insert(0, "$TESTBED/scripts")
from perf_model import ${MAKE_PROF} as make_profiles, save_profiles, ${DEPS_VAR} as DEPS

raw = open("$OUTPUT_DIR/profiling_raw.txt").read()
ops = ["${OP0}","G2rSparseMcoOffLoadOp","G2rSparseMcoMaskLoadOp",
       "G2sSparseMcoValLoadOp","G2sMatrixBLoadOp","S2sRestoreMatrixAOp",
       "S2rAValLoadOp","S2rBValLoadOp","CalculateOp"]
prof = defaultdict(lambda: defaultdict(list))
sand = defaultdict(lambda: defaultdict(list))
for line in raw.splitlines():
    p = line.split()
    if len(p) < 6: continue
    try:
        if p[0] == "PROFILE" and int(p[2]) > 0: prof[p[4]][int(p[1])].append(int(p[5]))
        elif p[0] == "SANDWICH" and int(p[2]) > 0: sand[int(p[3])][int(p[1])].append(int(p[5]))
    except: pass
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
        sw = stat.median(v); ta = stat.median(prof[pn].get(blk,[0]))
        tb = stat.median(prof[cn].get(blk,[0])) if cn else 0
        tails.append(max(sw - ta - tb, 0))
    t_tail[pn] = stat.mean(tails) if tails else 0
profiles = make_profiles()
for op in ops:
    if op in t_issue: profiles[op].T_issue = t_issue[op]
    if op in t_tail: profiles[op].T_tail = t_tail[op]
save_profiles(profiles, "$OUTPUT_DIR/op_profiles.json", dependencies=DEPS,
              metadata={"arch": "$ARCH"})
for op in ops: print(f"  {op:<30s} T_issue={t_issue[op]:7.0f}  T_tail={t_tail[op]:7.0f}")
PYEOF
fi
echo ""

# ===== Step 3: Compile + Time =====
if [ "$SKIP_TIMING" -eq 1 ]; then
    echo ">>> Step 3: Skipping compile+time"
else
    echo ">>> Step 3: Compile+time $TOTAL plans ($COMPILE_JOBS workers, GPUs: $GPUS)"

    # Clean empty timing files from previous failed runs
    find "$OUTPUT_DIR/outputs" -name "*.txt" -empty -delete 2>/dev/null

    # Ensure common .o
    for src in src/utils.cu src/mmio.cu src/mmio_highlevel.cu; do
        obj="$OUTPUT_DIR/obj/$(basename ${src%.cu}.o)"
        [ -f "$obj" ] || nvcc $NVCC_FLAGS $INC -dc -o "$obj" "$src" 2>/dev/null
    done
    [ -f "$OUTPUT_DIR/obj/main.o" ] || nvcc $NVCC_FLAGS $INC -dc -o "$OUTPUT_DIR/obj/main.o" main.cu 2>/dev/null
    COMMON="$OUTPUT_DIR/obj/utils.o $OUTPUT_DIR/obj/mmio.o $OUTPUT_DIR/obj/mmio_highlevel.o"

    # Sanity
    P0=$(ls "$PLANS_DIR"/plan_0000.inc)
    sed "s|#include \"kernel.inc\"|#include \"$P0\"|" host_program.cu > "$OUTPUT_DIR/obj/_s.cu"
    nvcc $NVCC_FLAGS $INC -dc -o "$OUTPUT_DIR/obj/_s.o" "$OUTPUT_DIR/obj/_s.cu" 2>/dev/null
    nvcc $NVCC_FLAGS -o "$OUTPUT_DIR/obj/_s" "$OUTPUT_DIR/obj/_s.o" "$OUTPUT_DIR/obj/main.o" $COMMON $LINK 2>/dev/null
    CUDA_VISIBLE_DEVICES=$FIRST_GPU "$OUTPUT_DIR/obj/_s" $ARGS 2>/dev/null | grep "mykernel" \
        && echo "  Sanity OK" || { echo "  SANITY FAIL"; exit 1; }
    rm -f "$OUTPUT_DIR/obj/_s" "$OUTPUT_DIR/obj/_s.o" "$OUTPUT_DIR/obj/_s.cu"

    # Timing workers (directory-based queue)
    mkdir -p "$OUTPUT_DIR/ready"
    timing_worker() {
        local GID=$1 COUNT=0
        while true; do
            local R=$(ls "$OUTPUT_DIR/ready/"*.ready 2>/dev/null | head -1)
            if [ -z "$R" ]; then
                [ -f "$OUTPUT_DIR/ready/ALL_DONE" ] && break
                sleep 0.1; continue
            fi
            local PID=$(basename "$R" .ready)
            rm -f "$R"
            local BIN="$OUTPUT_DIR/obj/_b${PID}"
            if [ -f "$BIN" ]; then
                CUDA_VISIBLE_DEVICES=$GID "$BIN" $ARGS > "$OUTPUT_DIR/outputs/plan_${PID}_timing.txt" 2>/dev/null
                [ "$KEEP_BINARIES" -eq 0 ] && rm -f "$BIN"
            fi
            COUNT=$((COUNT+1))
            [ $((COUNT % 200)) -eq 0 ] && echo "  GPU $GID: $COUNT timed"
        done
        echo "  GPU $GID done ($COUNT)"
    }
    export -f timing_worker
    export OUTPUT_DIR ARGS KEEP_BINARIES

    TPIDS=""
    for G in "${GPU_ARR[@]}"; do timing_worker "$G" & TPIDS="$TPIDS $!"; done

    # Write compile worker script (avoids export -f portability issues)
    cat > "$OUTPUT_DIR/obj/_compile_one.sh" << 'COMPILE_SCRIPT'
#!/bin/bash
PLAN="$1"
PID=$(basename "$PLAN" | sed 's/plan_//;s/.inc//')
[ -s "$OUTPUT_DIR/outputs/plan_${PID}_timing.txt" ] && exit 0
CU="$OUTPUT_DIR/obj/_c${PID}.cu"
OBJ="$OUTPUT_DIR/obj/_c${PID}.o"
BIN="$OUTPUT_DIR/obj/_b${PID}"
sed "s|#include \"kernel.inc\"|#include \"$PLAN\"|" "$TESTBED/host_program.cu" > "$CU"
nvcc $NVCC_FLAGS $INC -dc -o "$OBJ" "$CU" 2>/dev/null || { rm -f "$CU" "$OBJ"; exit 1; }
nvcc $NVCC_FLAGS -o "$BIN" "$OBJ" "$OUTPUT_DIR/obj/main.o" $COMMON $LINK 2>/dev/null
rm -f "$CU" "$OBJ"
[ -f "$BIN" ] && touch "$OUTPUT_DIR/ready/${PID}.ready"
COMPILE_SCRIPT
    chmod +x "$OUTPUT_DIR/obj/_compile_one.sh"
    export TESTBED OUTPUT_DIR NVCC_FLAGS INC LINK COMMON

    ls "$PLANS_DIR"/plan_*.inc | xargs -P $COMPILE_JOBS -I{} bash "$OUTPUT_DIR/obj/_compile_one.sh" {}
    touch "$OUTPUT_DIR/ready/ALL_DONE"
    for P in $TPIDS; do wait $P; done
    rm -rf "$OUTPUT_DIR/ready"

    TIMED=$(find "$OUTPUT_DIR/outputs" -name "*.txt" -size +0 | wc -l)
    echo "  Timed: $TIMED / $TOTAL"
fi
echo ""

# ===== Step 4: Simulate + combined_results.json =====
echo ">>> Step 4: Building combined_results.json"

if [ "$FORMAT" = "acc" ]; then
    MAKE_PROF="make_acc_spmm_profiles"; DEPS_VAR="ACC_SPMM_DEPENDENCIES"
else
    MAKE_PROF="make_bitbsr_spmm_profiles"; DEPS_VAR="BITBSR_SPMM_DEPENDENCIES"
fi

python3 << PYEOF
import sys, json, re, os, math
sys.path.insert(0, "$TESTBED/scripts")
from perf_model import ${MAKE_PROF} as make_profiles, load_profiles, ${DEPS_VAR} as DEPS, Pipeline, predict_steady_state_ii

prof_path = "$OUTPUT_DIR/op_profiles.json"
profiles = load_profiles(prof_path, defaults=make_profiles()) if os.path.exists(prof_path) else make_profiles()
manifest = json.load(open("$MANIFEST"))
id_to_entry = {e["id"]: e for e in manifest}

timing = {}
out_dir = "$OUTPUT_DIR/outputs"
if os.path.isdir(out_dir):
    for f in os.listdir(out_dir):
        if not f.endswith("_timing.txt"): continue
        pid = int(f.replace("plan_","").replace("_timing.txt",""))
        for line in open(f"{out_dir}/{f}"):
            m = re.match(r"mykernel_time:\s+([\d.]+)", line)
            if m: timing[pid] = float(m.group(1)) * 1000; break

sim = {}
for pid in (timing or id_to_entry):
    if pid in id_to_entry:
        try: sim[pid] = predict_steady_state_ii(Pipeline(stages=id_to_entry[pid]["stages"], shifts=id_to_entry[pid]["shifts"]), profiles, DEPS)
        except: pass

pids = sorted(timing) if timing else sorted(id_to_entry)
plans = [{"id":p, "stages":id_to_entry.get(p,{}).get("stages"), "shifts":id_to_entry.get(p,{}).get("shifts"),
          "kernel_time_us":timing.get(p), "predicted_ii":sim.get(p)} for p in pids]

# Metrics
metrics = {}
valid = [p for p in plans if p["predicted_ii"] is not None and p["kernel_time_us"] is not None]
n = len(valid)
if n >= 2:
    ro = sorted(range(n), key=lambda i: valid[i]["kernel_time_us"])
    so = sorted(range(n), key=lambda i: valid[i]["predicted_ii"])
    rr = [0]*n; sr = [0]*n
    for r,i in enumerate(ro): rr[i]=r
    for r,i in enumerate(so): sr[i]=r
    c=d=0
    for i in range(n):
        for j in range(i+1,n):
            p=sr[i]-sr[j]; t=rr[i]-rr[j]
            if p*t>0: c+=1
            elif p*t<0: d+=1
    metrics["kendall_tau"] = (c-d)/(c+d) if c+d else 0
    metrics["spearman_rho"] = 1 - 6*sum((s-r)**2 for s,r in zip(sr,rr))/(n*(n*n-1))
    mx = max(p["kernel_time_us"] for p in valid)
    rel = [mx - p["kernel_time_us"] for p in valid]
    dcg = sum(rel[so[i]]/math.log2(i+2) for i in range(n))
    io = sorted(range(n), key=lambda i: rel[i], reverse=True)
    idcg = sum(rel[io[i]]/math.log2(i+2) for i in range(n))
    metrics["ndcg"] = dcg/idcg if idcg else 0
    rm = {i:r+1 for r,i in enumerate(ro)}
    for k in [1,5,10,50]:
        if k<=n: metrics[f"top{k}_best_real_rank"] = min(rm[i] for i in so[:k])
    print(f"  Tau={metrics['kendall_tau']:.4f}  Rho={metrics['spearman_rho']:.4f}  NDCG={metrics['ndcg']:.4f}")
    for k in [1,5,10,50]:
        key=f"top{k}_best_real_rank"
        if key in metrics: print(f"  Top-{k} best real rank: {metrics[key]}")

op_prof = json.loads(open(prof_path).read()).get("op_profiles",{}) if os.path.exists(prof_path) else {}
json.dump({"metadata":{"format":"fp32_$FORMAT","arch":"$ARCH","gpu":"$GPU_NAME","matrix":"M=${M} K=${K} N=${N} sparsity=${SPARSITY}",
    "n_plans":len(plans),"n_timed":len(timing)}, "metrics":metrics, "op_profiles":op_prof, "plans":plans},
    open("$OUTPUT_DIR/combined_results.json","w"), indent=2)

if timing:
    bp = min(timing, key=timing.get)
    print(f"  Best: plan_{bp} = {timing[bp]:.1f} us, Worst: {max(timing.values()):.1f} us")
print(f"  Saved $OUTPUT_DIR/combined_results.json ({len(plans)} plans)")
PYEOF

echo ""
echo "============================================================"
echo " Done! Results: $OUTPUT_DIR/combined_results.json"
echo "============================================================"
