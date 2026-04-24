#!/usr/bin/env python3
"""
search_best_plan.py — End-to-end simulator-guided plan search
==============================================================

Fully automated pipeline:
  1. Generate all valid pipeline plans (codegen -> plan_XXXX.inc)
  2. Run profiling kernel -> op_profiles.json
  3. Simulate all plans -> rank by predicted II
  4. Compile & time top-N candidates on GPU
  5. Report the best plan

Output directory structure:
  <output_dir>/
    plans/              plan_XXXX.inc files
    profiles/           op_profiles.json, profiling_raw.txt, profiling binary
    simulation/         manifest.json, ranking.json
    timing/
      bin/              plan_XXXX_timing binaries
      obj/              precompiled objects
      outputs/          plan_XXXX_timing.txt
    best_plan.json      final result

Usage (from testbed directory):
  python scripts/search_best_plan.py -o search_results --top-n 50 \\
      -M 1024 -K 1024 --sparsity 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


TESTBED = Path(__file__).resolve().parent.parent
SCRIPTS = TESTBED / "scripts"

NVCC_FLAGS = (
    "-std=c++17 -O3 -m64 -g -lineinfo -Xcompiler -fopenmp-simd "
    "--expt-relaxed-constexpr -ftemplate-backtrace-limit=0 -Df32"
)
INCLUDE_FLAGS = (
    f"-I{TESTBED} -I{TESTBED}/src -I{TESTBED}/cutlass/include "
    f"-I{TESTBED}/cutlass/examples/common -I{TESTBED}/cutlass/tools/util/include"
)
LINK_FLAGS = "-lcusparse -lcublas"


def run(cmd, cwd=None, quiet=False):
    if not quiet:
        print(f"  $ {cmd[:120]}{'...' if len(cmd)>120 else ''}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd or TESTBED)
    if r.returncode != 0 and not quiet:
        print(f"    FAILED (exit {r.returncode}): {r.stderr[:300]}")
    return r


# ---------------------------------------------------------------------------
# Step 1: Generate plans
# ---------------------------------------------------------------------------

def generate_plans(output_dir: Path, args):
    plans_dir = output_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: Generate pipeline plans ===")
    sys.path.insert(0, str(SCRIPTS))
    from plan_searcher import bitbsr
    from sparsene.op_gen.nvir.plan import apply_pipeline
    from sparsene.op_gen.nvir.pipeline.pipeline_planner import (
        enumerate_pipeline_plans, NeighborDependencyValidator,
    )
    from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence
    from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator

    main_loop_op, program = bitbsr()
    op_graph = construct_graph_from_op_sequence(main_loop_op.body)

    plans = enumerate_pipeline_plans(
        main_loop_op,
        NeighborDependencyValidator(op_graph),
        min_nstages=args.min_stages, max_nstages=args.max_stages,
        min_ops_per_stage=1, max_ops_per_stage=args.max_ops_per_stage,
        min_shift=1, max_shift=args.max_shift,
    )
    print(f"  Enumerated {len(plans)} valid plans")

    manifest = []
    t0 = time.time()
    for idx, plan in enumerate(plans):
        new_loop, new_program = bitbsr()
        apply_pipeline(new_loop, plan)
        with open(plans_dir / f"plan_{idx:04d}.inc", "w") as f:
            f.write(NvIrCodeGenerator().dump_nvop_program(new_program))
        manifest.append({
            "id": idx,
            "stages": [[op.name for op in stage.ops] for stage in plan.stages],
            "shifts": list(plan.shifts),
        })
    print(f"  Codegen: {time.time()-t0:.1f}s")

    sim_dir = output_dir / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    with open(sim_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved {len(manifest)} plans")
    return manifest


# ---------------------------------------------------------------------------
# Step 2: Profile
# ---------------------------------------------------------------------------

def compile_common(obj_dir: Path, arch: str):
    """Compile common objects (shared across profiling + timing).
    Always recompiles to avoid stale .o from different nvcc/flags."""
    nvcc = f"{NVCC_FLAGS} -arch={arch}"
    for src in ["src/utils.cu", "src/mmio.cu", "src/mmio_highlevel.cu"]:
        obj = obj_dir / Path(src).with_suffix(".o").name
        run(f"nvcc {nvcc} {INCLUDE_FLAGS} -dc -o {obj} {TESTBED / src}")
    main_obj = obj_dir / "main.o"
    if True:  # always recompile
        run(f"nvcc {nvcc} {INCLUDE_FLAGS} -dc -o {main_obj} {TESTBED / 'main.cu'}")


def run_profiling(output_dir: Path, args):
    print("\n=== Step 2: Run profiling kernel ===")
    prof_dir = output_dir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = output_dir / "timing" / "obj"
    obj_dir.mkdir(parents=True, exist_ok=True)

    nvcc = f"{NVCC_FLAGS} -arch={args.arch}"
    compile_common(obj_dir, args.arch)

    # Build profiling binary
    prof_obj = obj_dir / "host_program_profiling.o"
    run(f"nvcc {nvcc} {INCLUDE_FLAGS} -dc -o {prof_obj} {TESTBED / 'host_program_profiling.cu'}")
    prof_bin = prof_dir / "profiling"
    main_obj = obj_dir / "main.o"
    common = " ".join(str(obj_dir / f) for f in ["utils.o", "mmio.o", "mmio_highlevel.o"])
    run(f"nvcc {nvcc} -o {prof_bin} {prof_obj} {main_obj} {common} {LINK_FLAGS}")

    # Run (use first GPU if --gpus specified)
    kernel_args = f"-M {args.M} -K {args.K} -N {args.N} -sparsity {args.sparsity} -mtx_flag 0 -ncu 1"
    gpu_prefix = f"CUDA_VISIBLE_DEVICES={args.gpu_ids[0]}" if args.gpu_ids else ""
    r = run(f"{gpu_prefix} PROF_NBLOCKS=64 {prof_bin} {kernel_args}")

    with open(prof_dir / "profiling_raw.txt", "w") as f:
        f.write(r.stdout)
        f.write(r.stderr)

    _parse_profiling_output(r.stdout, prof_dir / "op_profiles.json")
    print(f"  Saved op_profiles.json")


def _parse_profiling_output(raw: str, out_path: Path):
    import statistics as stat
    sys.path.insert(0, str(SCRIPTS))
    from perf_model import make_bitbsr_spmm_profiles, save_profiles, BITBSR_SPMM_DEPENDENCIES

    ops = ["G2rSparseIndexLoadOp", "G2rSparseMcoOffLoadOp", "G2rSparseMcoMaskLoadOp",
           "G2sSparseMcoValLoadOp", "G2sMatrixBLoadOp", "S2sRestoreMatrixAOp",
           "S2rAValLoadOp", "S2rBValLoadOp", "CalculateOp"]

    prof = defaultdict(lambda: defaultdict(list))
    sand = defaultdict(lambda: defaultdict(list))
    for line in raw.splitlines():
        p = line.split()
        if len(p) >= 6:
            if p[0] == "PROFILE" and int(p[2]) > 0:
                prof[p[4]][int(p[1])].append(int(p[5]))
            elif p[0] == "SANDWICH" and int(p[2]) > 0:
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
    save_profiles(profiles, str(out_path), dependencies=BITBSR_SPMM_DEPENDENCIES,
                  metadata={"source": "profiling kernel, 64-block average"})


# ---------------------------------------------------------------------------
# Step 3: Simulate & rank
# ---------------------------------------------------------------------------

def simulate_and_rank_all(output_dir: Path, manifest, top_n: int):
    print(f"\n=== Step 3: Simulate {len(manifest)} plans ===")
    sys.path.insert(0, str(SCRIPTS))
    from perf_model import (
        Pipeline, predict_steady_state_ii, BITBSR_SPMM_DEPENDENCIES,
        make_bitbsr_spmm_profiles, load_profiles,
    )

    prof_path = output_dir / "profiles" / "op_profiles.json"
    profiles = (load_profiles(str(prof_path), defaults=make_bitbsr_spmm_profiles())
                if prof_path.exists() else make_bitbsr_spmm_profiles())

    results = []
    for entry in manifest:
        pipeline = Pipeline(stages=entry["stages"], shifts=entry["shifts"])
        try:
            ii = predict_steady_state_ii(pipeline, profiles, BITBSR_SPMM_DEPENDENCIES)
            results.append((entry["id"], ii))
        except Exception:
            pass
    results.sort(key=lambda x: x[1])

    sim_dir = output_dir / "simulation"
    with open(sim_dir / "ranking.json", "w") as f:
        json.dump([{"id": pid, "predicted_ii": ii} for pid, ii in results], f, indent=2)

    selected = results[:top_n]
    print(f"  Top-{top_n} selected (best II = {selected[0][1]:.0f})")
    return selected


# ---------------------------------------------------------------------------
# Step 4: Compile & time top-N
# ---------------------------------------------------------------------------

def compile_and_time(output_dir: Path, selected, args):
    """Compile and time selected plans with pipelined execution."""
    import queue
    import threading

    print(f"\n=== Step 4: Compile & time {len(selected)} plans ===")

    plans_dir = output_dir / "plans"
    bin_dir = output_dir / "timing" / "bin"
    obj_dir = output_dir / "timing" / "obj"
    out_dir = output_dir / "timing" / "outputs"
    bin_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    nvcc = f"{NVCC_FLAGS} -arch={args.arch}"
    compile_common(obj_dir, args.arch)
    main_obj = obj_dir / "main.o"
    common = " ".join(str(obj_dir / f) for f in ["utils.o", "mmio.o", "mmio_highlevel.o"])

    # Read host_program.cu template once
    with open(TESTBED / "host_program.cu") as f:
        host_src_template = f.read()

    kernel_args = (f"-M {args.M} -K {args.K} -N {args.N} -sparsity {args.sparsity} "
                   f"-mtx_flag 0 -warmup {args.warmup} -repeat {args.repeat}")

    # Event log for timing analysis
    log_path = output_dir / "timing" / "event_log.jsonl"
    log_lock = threading.Lock()
    log_file = open(log_path, "a")

    def log_event(pid, event, **extra):
        entry = {"pid": pid, "event": event, "ts": time.time(), **extra}
        with log_lock:
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

    # Queue: compiled binaries ready for timing (pid, binary_path)
    ready_q: queue.Queue = queue.Queue()
    compile_jobs = getattr(args, 'compile_jobs', 45)

    # --- Compiler worker ---
    def compile_one(pid):
        kernel = plans_dir / f"plan_{pid:04d}.inc"
        binary = bin_dir / f"plan_{pid:04d}_timing"
        if binary.exists():
            log_event(pid, "compile_cached")
            ready_q.put((pid, binary))
            return
        if not kernel.exists():
            return

        log_event(pid, "compile_start")

        tmp_cu = obj_dir / f"_tmp_{pid:04d}.cu"
        src = host_src_template.replace(
            '#include "kernel.inc"', f'#include "{kernel.resolve()}"')
        with open(tmp_cu, "w") as f:
            f.write(src)

        tmp_obj = obj_dir / f"_tmp_{pid:04d}.o"
        r1 = run(f"nvcc {nvcc} {INCLUDE_FLAGS} -dc -o {tmp_obj} {tmp_cu}", quiet=True)
        if r1.returncode != 0:
            log_event(pid, "compile_fail")
            tmp_cu.unlink(missing_ok=True)
            return
        r2 = run(f"nvcc {nvcc} -o {binary} {tmp_obj} {main_obj} {common} {LINK_FLAGS}", quiet=True)
        tmp_cu.unlink(missing_ok=True)
        tmp_obj.unlink(missing_ok=True)
        if r2.returncode == 0:
            log_event(pid, "compile_end")
            ready_q.put((pid, binary))
        else:
            log_event(pid, "link_fail")

    # --- Compiler thread pool ---
    compile_count = [0]
    compile_lock_count = threading.Lock()

    def compile_worker():
        while True:
            try:
                pid = compile_q.get_nowait()
            except queue.Empty:
                break
            # Throttle: don't let too many binaries pile up on disk
            while ready_q.qsize() > 200:
                import time as _time
                _time.sleep(1)
            compile_one(pid)
            with compile_lock_count:
                compile_count[0] += 1
                done = compile_count[0]
            if done % 200 == 0:
                print(f"  Compiled {done}/{len(selected)}...")
            compile_q.task_done()

    compile_q: queue.Queue = queue.Queue()
    for pid, _ in selected:
        compile_q.put(pid)

    threads = []
    for _ in range(min(compile_jobs, len(selected))):
        t = threading.Thread(target=compile_worker, daemon=True)
        t.start()
        threads.append(t)

    # --- Timing workers (multi-GPU if --gpus specified) ---
    gpu_ids = getattr(args, 'gpu_ids', None) or [None]
    n_gpu = len(gpu_ids)

    results = {}
    results_lock = threading.Lock()
    timed_count = [0]
    total = len(selected)
    compile_done = threading.Event()

    def timing_worker(gpu_id):
        gpu_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id}" if gpu_id is not None else ""
        while True:
            try:
                pid, binary = ready_q.get(timeout=2.0)
            except queue.Empty:
                if compile_done.is_set() and ready_q.empty():
                    break
                continue

            timing_file = out_dir / f"plan_{pid:04d}_timing.txt"
            if timing_file.exists():
                t_val = _read_timing(timing_file)
                if t_val is not None:
                    with results_lock:
                        results[pid] = t_val
                        timed_count[0] += 1
                    log_event(pid, "time_cached", time_us=t_val, gpu=gpu_id)
                    binary.unlink(missing_ok=True)
                    continue

            log_event(pid, "time_start", gpu=gpu_id)
            r = run(f"{gpu_prefix} {binary} {kernel_args}".strip(), quiet=True)
            log_event(pid, "time_end", gpu=gpu_id)

            with open(timing_file, "w") as f:
                f.write(r.stdout)
            t_val = _parse_timing(r.stdout)
            if t_val is not None:
                with results_lock:
                    results[pid] = t_val
                log_event(pid, "time_result", time_us=t_val, gpu=gpu_id)

            with results_lock:
                timed_count[0] += 1
                n_done = timed_count[0]

            binary.unlink(missing_ok=True)

            if n_done % 200 == 0:
                best_so_far = min(results.values()) if results else 0
                print(f"  Timed {n_done}/{total}, best so far: {best_so_far:.1f} us")

    timing_threads = []
    gpu_label = ",".join(str(g) for g in gpu_ids if g is not None) or "default"
    print(f"  Timing GPUs: [{gpu_label}] ({n_gpu} workers)")
    for gid in gpu_ids:
        t = threading.Thread(target=timing_worker, args=(gid,), daemon=True)
        t.start()
        timing_threads.append(t)

    for t in threads:
        t.join()
    compile_done.set()

    for t in timing_threads:
        t.join()
    log_file.close()

    print(f"  Compiled {compile_count[0]}/{total}")
    print(f"  Timed {len(results)}/{total}")
    if results:
        print(f"  Best: {min(results.values()):.1f} us")
    print(f"  Event log: {log_path}")
    return results


def _read_timing(path: Path):
    with open(path) as f:
        for line in f:
            m = re.match(r"KERNEL_TIME_US:\s+([\d.]+)", line)
            if m: return float(m.group(1))
            m = re.match(r"mykernel_time:\s+([\d.]+)", line)
            if m: return float(m.group(1)) * 1000
    return None


def _parse_timing(stdout: str):
    for line in stdout.splitlines():
        m = re.match(r"mykernel_time:\s+([\d.]+)", line)
        if m: return float(m.group(1)) * 1000
        m = re.match(r"KERNEL_TIME_US:\s+([\d.]+)", line)
        if m: return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Step 5: Report
# ---------------------------------------------------------------------------

def report(output_dir: Path, manifest, selected, timing):
    print(f"\n{'='*60}")
    if not timing:
        print("  No timing results!")
        return

    by_id = {e["id"]: e for e in manifest}
    best_pid = min(timing, key=lambda p: timing[p])
    best_time = timing[best_pid]
    best_entry = by_id.get(best_pid, {})

    result = {
        "best_plan_id": best_pid,
        "best_kernel_time_us": best_time,
        "stages": best_entry.get("stages", []),
        "shifts": best_entry.get("shifts", []),
        "plans_evaluated": len(timing),
        "top10": [{"id": pid, "time_us": t}
                  for pid, t in sorted(timing.items(), key=lambda x: x[1])[:10]],
    }
    with open(output_dir / "best_plan.json", "w") as f:
        json.dump(result, f, indent=2)

    stages_str = " | ".join([",".join(s) for s in best_entry.get("stages", [])])
    print(f"  Best plan: {best_pid:04d}  ({best_time:.1f} us)")
    print(f"  Stages: [{stages_str}]")
    print(f"  Shifts: {best_entry.get('shifts', [])}")
    print(f"  Evaluated: {len(timing)} plans")

    ranked = sorted(timing.items(), key=lambda x: x[1])
    print(f"\n  Top-5:")
    for i, (pid, t) in enumerate(ranked[:5]):
        print(f"    #{i}: plan_{pid:04d}  {t:.1f} us")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Simulator-guided plan search")
    p.add_argument("-o", "--output-dir", default="search_results")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("-M", type=int, default=1024)
    p.add_argument("-K", type=int, default=1024)
    p.add_argument("--N-dim", type=int, default=64, dest="N")
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--min-stages", type=int, default=2)
    p.add_argument("--max-stages", type=int, default=3)
    p.add_argument("--max-ops-per-stage", type=int, default=4)
    p.add_argument("--max-shift", type=int, default=3)
    p.add_argument("--arch", default="sm_89")
    p.add_argument("--compile-jobs", type=int, default=45)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--gpus", default=None,
                   help="GPU IDs for parallel timing, e.g. '4,5,6,7' or '4-7'.")
    args = p.parse_args()

    # Parse --gpus into a list of GPU IDs
    args.gpu_ids = None
    if args.gpus:
        gpu_ids = []
        for part in args.gpus.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                gpu_ids.extend(range(int(lo), int(hi) + 1))
            else:
                gpu_ids.append(int(part))
        args.gpu_ids = gpu_ids
        print(f"Using GPUs: {gpu_ids}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1
    manifest_path = output_dir / "simulation" / "manifest.json"
    if args.skip_generate and manifest_path.exists():
        manifest = json.load(open(manifest_path))
        print(f"Loaded {len(manifest)} plans from {manifest_path}")
    else:
        manifest = generate_plans(output_dir, args)

    # Step 2
    if not args.skip_profile:
        run_profiling(output_dir, args)

    # Step 3
    selected = simulate_and_rank_all(output_dir, manifest, args.top_n)

    # Step 4
    timing = compile_and_time(output_dir, selected, args)

    # Step 5
    report(output_dir, manifest, selected, timing)


if __name__ == "__main__":
    main()
