"""
Multi-Pipe Performance Model for SpMM Pipeline Plan Selection
=============================================================

Models GPU SM execution as a set of concurrent hardware resource pipes.
Each op has a T_issue (warp-busy time) and T_tail (background execution
after issue). Data dependencies and pipe conflicts constrain scheduling.
GMEM bandwidth contention between concurrent in-flight ops is modeled via
a pairwise contention table derived from profiling.

Terminology
-----------
- T_issue : cycles the warp spends issuing an op's instructions, including
            any internal scoreboard stalls within the op. Measured by
            clock() before/after the op's call.
- T_tail  : cycles the op's background execution continues after the warp
            has finished issuing. T_exec = T_issue + T_tail.
- T_exec  : T_issue + T_tail. The op's result is available T_exec cycles
            after the op starts.
- pipe    : an independent hardware execution unit (tensor core, async copy
            engine, etc.). Multiple pipes can be active concurrently.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Hardware pipe definitions
# ---------------------------------------------------------------------------

class Pipe(IntFlag):
    """Hardware resource pipes on an Ampere SM.

    Each op uses a subset of these pipes. Pipes are tracked independently;
    an op cannot start executing on a pipe until that pipe is free.
    GMEM_BW is a virtual pipe used only for bandwidth contention accounting,
    not for hardware conflict detection.
    """
    ASYNC_COPY  = auto()  # cp.async engine (background GMEM→SMEM transfer)
    LDST        = auto()  # Synchronous load/store unit (ld.global, st.global)
    TENSOR_CORE = auto()  # MMA unit (mma.sync / HMMA instructions)
    CUDA_CORE   = auto()  # INT/FP ALU (arithmetic, address computation, popcll)
    SMEM_READ   = auto()  # Shared memory read (ldmatrix, ld.shared)
    SMEM_WRITE  = auto()  # Shared memory write (st.shared, cp.async landing)
    GMEM_BW     = auto()  # Virtual: GMEM bandwidth (for contention tracking only)

    # Convenience alias — backward compat for code that uses Pipe.SMEM
    @classmethod
    @property
    def SMEM(cls) -> "Pipe":
        return cls.SMEM_READ | cls.SMEM_WRITE

# Pipes that represent real hardware units (used for conflict detection)
HARDWARE_PIPES = (Pipe.ASYNC_COPY | Pipe.LDST | Pipe.TENSOR_CORE | Pipe.CUDA_CORE
                  | Pipe.SMEM_READ | Pipe.SMEM_WRITE)

# ---------------------------------------------------------------------------
# Profiling metadata — describes HOW to profile each op
# ---------------------------------------------------------------------------

class OutputKind(Enum):
    """Where the op's result materializes. Determines how to force a
    scoreboard dependency when measuring T_exec.

    REGISTER   — result lands in registers (ld.global, ldmatrix, MMA acc).
                 Force T_exec by reading the output register.
    SMEM_ASYNC — result lands in SMEM via cp.async.  Force T_exec by
                 __pipeline_commit() + __pipeline_wait_prior(0) + __syncthreads()
                 then reading the SMEM location.
    SMEM_SYNC  — result lands in SMEM via st.shared (compute-heavy ops).
                 Force T_exec by __syncthreads() + reading SMEM.
    NONE       — op has no consumable output (e.g., a store op).
    """
    REGISTER   = "register"
    SMEM_ASYNC = "smem_async"
    SMEM_SYNC  = "smem_sync"
    NONE       = "none"


class IssueKind(Enum):
    """How the warp interacts with this op at issue time.

    ASYNC_CP         — issues cp.async instructions; warp moves on immediately
                       after issue. T_tail is large (GMEM transfer latency).
    SYNC_GMEM_LOAD   — issues ld.global instructions; warp moves on without
                       stalling (no RAW dep at issue), but the result register
                       isn't ready for ~200-400 cycles. T_tail = GMEM latency.
    COMPUTE_HEAVY    — warp-busy throughout (internal deps chain). T_issue
                       dominates; T_tail is small (last SMEM store latency).
    SMEM_LOAD        — issues ldmatrix / ld.shared; warp moves on quickly.
                       T_tail = SMEM latency (~20-30 cycles).
    TENSOR_CORE      — issues mma.sync instructions; warp issues quickly,
                       tensor core computes in background. T_tail = MMA latency.
    """
    ASYNC_CP       = "async_cp"
    SYNC_GMEM_LOAD = "sync_gmem_load"
    COMPUTE_HEAVY  = "compute_heavy"
    SMEM_LOAD      = "smem_load"
    TENSOR_CORE    = "tensor_core"


@dataclass
class OpProfilingMeta:
    """Metadata that describes HOW to profile a specific op.

    This is attached to each OpProfile and drives:
      1. T_issue measurement strategy (clock around op call).
      2. T_exec measurement strategy (force dependency based on output_kind).
      3. Contention pairing (ops sharing a contention_group are measured pairwise).
      4. Profiling setup requirements (input_deps, needs_smem_setup).

    Fields
    ------
    output_kind : OutputKind
        Where the result lives — determines how to force a scoreboard dep for
        T_exec measurement.
    issue_kind : IssueKind
        How the warp behaves during issue — used to select the right
        profiling wrapper (e.g., async ops need pipeline state management).
    contention_groups : frozenset of str
        Set of shared-resource group names this op belongs to.
        Ops sharing ANY contention group are measured pairwise for slowdown.
        Typical groups: "gmem_bulk" (large cp.async), "gmem_scalar" (small
        ld.global), "smem_rw" (competing SMEM reads/writes).
        Empty set means no contention profiling needed.
    input_deps : tuple of str
        Names of ops whose outputs must be available before this op can
        execute during profiling. Used to set up the correct profiling context
        (e.g., Op5 needs Op1's SMEM output pre-populated).
    needs_smem_setup : bool
        If True, profiling this op requires pre-populating SMEM buffers with
        representative data (e.g., Op7 needs smem_A_res filled by Op6).
    T_tail_hint : optional float
        If set, use this as a fixed T_tail estimate instead of measuring T_exec.
        Useful for ops with well-known architectural latencies (ldmatrix ~25 cyc,
        MMA ~32 cyc on Ampere).
    """
    output_kind: OutputKind
    issue_kind: IssueKind
    contention_groups: FrozenSet[str] = frozenset()
    input_deps: Tuple[str, ...] = ()
    needs_smem_setup: bool = False
    T_tail_hint: Optional[float] = None


# ---------------------------------------------------------------------------
# Op profile data structure
# ---------------------------------------------------------------------------

@dataclass
class OpProfile:
    """Profiling data for a single op, measured at SM level.

    Fields
    ------
    name : str
        Op identifier, must match keys in dependency lists.
    T_issue : float
        Warp-busy time in GPU clock cycles (from clock() instrumentation).
        Includes internal scoreboard stalls within the op.
    T_tail : float
        Background execution time after the warp finishes issuing, in cycles.
        T_exec = T_issue + T_tail. Op output available T_exec cycles after start.
    pipes : Pipe
        Bitmask of hardware pipes this op uses (may include GMEM_BW).
    bytes_gmem : int
        GMEM bytes transferred per invocation (0 if no GMEM access).
        Used for bandwidth contention modeling.
    profiling : optional OpProfilingMeta
        Metadata describing how to profile this op. If None, the op cannot be
        automatically profiled (manual measurement required).
    """
    name: str
    T_issue: float
    T_tail: float
    pipes: Pipe
    bytes_gmem: int = 0
    profiling: Optional[OpProfilingMeta] = None

    @property
    def T_exec(self) -> float:
        """Total execution latency: T_issue + T_tail."""
        return self.T_issue + self.T_tail

    @property
    def uses_gmem_bw(self) -> bool:
        return bool(self.pipes & Pipe.GMEM_BW)

    @property
    def hardware_pipes(self) -> Pipe:
        """Pipe flags for real hardware units (excluding GMEM_BW)."""
        return self.pipes & HARDWARE_PIPES


# ---------------------------------------------------------------------------
# Contention table
# ---------------------------------------------------------------------------

# contention_slowdown[op_a_name][op_b_name] = factor by which op_a's T_tail
# stretches when op_b has overlapping in-flight GMEM requests.
# 1.0 = no contention, >1.0 = slowdown.
ContentionTable = Dict[str, Dict[str, float]]


def no_contention_table() -> ContentionTable:
    """Returns an empty contention table (no slowdown for any pair)."""
    return {}


# ---------------------------------------------------------------------------
# Schedule entry (simulation output)
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    """Timing record for one op invocation in the simulated dispatch queue."""
    op_name: str
    iteration: int      # Loop iteration index this invocation belongs to
    issue_start: float  # When the warp starts issuing this op
    issue_end: float    # = issue_start + T_issue (warp free for next op)
    exec_start: float   # When the hardware pipe starts executing (>= issue_start)
    exec_end: float     # = exec_start + T_tail_adjusted (op result available)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# List of (producer_op_name, consumer_op_name) pairs for the same iteration.
# Cross-iteration dependencies (e.g., stage 0 of iter i+1 deps on stage 2 of iter i)
# are expressed implicitly through the dispatch queue ordering.
Dependencies = List[Tuple[str, str]]


def build_consumer_to_producers(dependencies: Dependencies) -> Dict[str, List[str]]:
    """Inverts dependency list into a consumer→[producers] map."""
    result: Dict[str, List[str]] = {}
    for producer, consumer in dependencies:
        result.setdefault(consumer, []).append(producer)
        result.setdefault(producer, [])  # ensure producer has an entry
    return result


# ---------------------------------------------------------------------------
# Core simulation: dispatch queue → schedule
# ---------------------------------------------------------------------------

def simulate_dispatch_queue(
    dispatch_queue: List[Tuple[str, int]],
    profiles: Dict[str, OpProfile],
    dependencies: Dependencies,
    contention_table: Optional[ContentionTable] = None,
) -> Tuple[float, List[ScheduleEntry]]:
    """Simulate execution of a dispatch queue using the multi-pipe model.

    The dispatch queue is a flat, ordered list of (op_name, iteration_index)
    tuples representing every op invocation across all iterations and pipeline
    stages in execution order.

    Scheduling rules
    ----------------
    1. **Warp issue serialization**: The warp has a single issue port. It
       issues op N only after it finishes issuing op N-1 (warp_issue_time).
    2. **Data dependencies**: An op cannot start issuing until all producer
       ops' results are available (producer.exec_end).
    3. **Pipe conflicts**: An op cannot start executing on a pipe until that
       pipe is free (pipe_free[pipe]).
    4. **GMEM bandwidth contention**: If a GMEM-accessing op starts while
       another GMEM op's transfer is still in flight, its T_tail is stretched
       by the worst-case slowdown from the contention table.

    Parameters
    ----------
    dispatch_queue : list of (op_name, iteration_index)
        Ordered sequence of op invocations. Generate via fill_dispatch_queue().
    profiles : dict of op_name → OpProfile
        Per-op profiling data (T_issue, T_tail, pipes, bytes_gmem).
    dependencies : list of (producer_name, consumer_name)
        Same-iteration data dependencies.
    contention_table : optional ContentionTable
        Pairwise GMEM slowdown factors. If None or empty, no contention applied.

    Returns
    -------
    t_end : float
        Total simulation time in cycles (end of last op's execution).
    schedule : list of ScheduleEntry
        Per-op timing details.
    """
    if contention_table is None:
        contention_table = {}

    consumer_to_producers = build_consumer_to_producers(dependencies)

    # Simulation state
    warp_issue_time: float = 0.0
    pipe_free: Dict[Pipe, float] = {p: 0.0 for p in Pipe}
    # (op_name, iteration) -> exec_end: when op's result is available
    op_exec_end: Dict[Tuple[str, int], float] = {}
    # In-flight GMEM ops: list of (exec_end, op_name) for contention tracking
    gmem_in_flight: List[Tuple[float, str]] = []

    schedule: List[ScheduleEntry] = []

    for op_name, iteration in dispatch_queue:
        p = profiles[op_name]

        # --- 1. Earliest issue start ---
        # Warp must be free, AND all data dependencies resolved.
        issue_start = warp_issue_time
        for producer in consumer_to_producers.get(op_name, []):
            key = (producer, iteration)
            if key in op_exec_end:
                issue_start = max(issue_start, op_exec_end[key])

        issue_end = issue_start + p.T_issue

        # --- 2. Execution timing ---
        # No pipe exclusion except TENSOR_CORE (one MMA at a time).
        exec_start = issue_start
        if p.pipes & Pipe.TENSOR_CORE:
            exec_start = max(exec_start, pipe_free.get(Pipe.TENSOR_CORE, 0.0))

        # --- 3. GMEM bandwidth contention adjustment ---
        T_tail_adj = p.T_tail
        if p.uses_gmem_bw and p.bytes_gmem > 0:
            gmem_in_flight = [(end, name) for (end, name) in gmem_in_flight
                              if end > exec_start]
            worst_slowdown = 1.0
            op_contention = contention_table.get(op_name, {})
            for _, other_name in gmem_in_flight:
                slowdown = op_contention.get(other_name, 1.0)
                worst_slowdown = max(worst_slowdown, slowdown)
            T_tail_adj = p.T_tail * worst_slowdown

        exec_end = max(issue_end, exec_start) + T_tail_adj

        # --- 4. Update simulation state ---
        warp_issue_time = issue_end

        # Only track TENSOR_CORE exclusion
        if p.pipes & Pipe.TENSOR_CORE:
            pipe_free[Pipe.TENSOR_CORE] = max(
                pipe_free.get(Pipe.TENSOR_CORE, 0.0), exec_end)

        op_exec_end[(op_name, iteration)] = exec_end

        if p.uses_gmem_bw and p.bytes_gmem > 0:
            gmem_in_flight.append((exec_end, op_name))

        schedule.append(ScheduleEntry(
            op_name=op_name,
            iteration=iteration,
            issue_start=issue_start,
            issue_end=issue_end,
            exec_start=exec_start,
            exec_end=exec_end,
        ))

    t_end = max((e.exec_end for e in schedule), default=0.0)
    return t_end, schedule


# ---------------------------------------------------------------------------
# Pipeline dispatch queue generation
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    """Representation of a software pipeline plan.

    Attributes
    ----------
    stages : list of list of str
        Each element is an ordered list of op names in that stage.
    shifts : list of int
        shifts[i] is the iteration distance between stage i and stage i+1.
        len(shifts) == len(stages) - 1.
    """
    stages: List[List[str]]
    shifts: List[int]

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def max_shift(self) -> int:
        return max(self.shifts) if self.shifts else 0

    @property
    def nbuf(self) -> int:
        """Number of pipeline buffers = max_shift + 1."""
        return self.max_shift + 1


def fill_dispatch_queue(
    k: int,
    pipeline: Pipeline,
) -> List[Tuple[str, int]]:
    """Generate the dispatch queue for a k-iteration software-pipelined loop.

    The dispatch queue is an ordered flat list of (op_name, iteration_index)
    covering the fill phase, steady-state loop iterations, and drain phase.

    Parameters
    ----------
    k : int
        Total number of loop iterations.
    pipeline : Pipeline
        Pipeline plan (stages and shifts).

    Returns
    -------
    list of (op_name, iteration_index)
    """
    dispatch_queue: List[Tuple[str, int]] = []
    fill_len = sum(pipeline.shifts) + pipeline.max_shift
    nbuf = pipeline.nbuf

    # Tracks next iteration index for each op (used to assign correct iter ids)
    pipeline_history: Dict[str, int] = {
        op: 0
        for stage in pipeline.stages
        for op in stage
    }

    def _next_id(op: str) -> int:
        idx = pipeline_history[op]
        pipeline_history[op] += 1
        return idx

    def short_pipeline_dispatch() -> None:
        """Handle the case where k <= fill_len (pipeline never reaches steady state)."""
        stages = pipeline.stages
        shifts = pipeline.shifts
        for i in range(k + sum(shifts)):
            for stage_idx, stage in enumerate(stages):
                shift_val = sum(shifts[:stage_idx])
                if 0 <= i - shift_val < k:
                    for op in stage:
                        dispatch_queue.append((op, i - shift_val))

    def fill_dispatch() -> None:
        """Fill/prolog: partial pipeline fill iterations."""
        for i in range(len(pipeline.stages)):
            nsteps = pipeline.shifts[i] if i < len(pipeline.shifts) else pipeline.max_shift
            for _ in range(nsteps):
                for stage in pipeline.stages[: i + 1]:
                    for op in stage:
                        dispatch_queue.append((op, _next_id(op)))

    def loop_step_dispatch() -> None:
        """One nbuf-sized chunk of steady-state loop body."""
        for _ in range(nbuf):
            for stage in pipeline.stages:
                for op in stage:
                    dispatch_queue.append((op, _next_id(op)))

    def remainder_dispatch(i: int) -> None:
        """Handle remaining iterations after the last full steady-state chunk."""
        remain = k - i
        if remain <= 0:
            return
        for _ in range(remain):
            for stage in pipeline.stages:
                for op in stage:
                    dispatch_queue.append((op, _next_id(op)))

    def empty_dispatch(i: int) -> None:
        """Drain/epilog: flush remaining in-flight stages."""
        remain = k - i
        if remain <= 0:
            return
        for stage_i in range(1, len(pipeline.stages)):
            nsteps = pipeline.shifts[stage_i - 1]
            for _ in range(nsteps):
                for stage in pipeline.stages[stage_i:]:
                    for op in stage:
                        dispatch_queue.append((op, _next_id(op)))

    if k <= fill_len:
        short_pipeline_dispatch()
    else:
        fill_dispatch()
        i = fill_len
        while i + nbuf <= k:
            loop_step_dispatch()
            i += nbuf
        remainder_dispatch(i)
        empty_dispatch(i)

    return dispatch_queue


# ---------------------------------------------------------------------------
# Pipeline performance prediction
# ---------------------------------------------------------------------------

# Empirically calibrated constants (GPU clock cycles on Ampere A100)
# Override these after measuring with barrier microbenchmarks.
SYNC_OVERHEAD_PER_STAGE_BOUNDARY: float = 0.0  # barrier overhead negligible vs pipeline benefit


def predict_pipeline_time(
    pipeline: Pipeline,
    k: int,
    profiles: Dict[str, OpProfile],
    dependencies: Dependencies,
    contention_table: Optional[ContentionTable] = None,
    sync_overhead_per_boundary: float = SYNC_OVERHEAD_PER_STAGE_BOUNDARY,
) -> float:
    """Predict the total execution time (in cycles) for a k-iteration pipeline.

    Generates the full dispatch queue, runs the multi-pipe simulation, and
    returns the predicted cycle count.

    Parameters
    ----------
    pipeline : Pipeline
        The software pipeline plan.
    k : int
        Number of loop iterations (the K-tile loop trip count).
    profiles : dict
        Per-op profiling data.
    dependencies : list of (producer, consumer)
        Same-iteration data dependencies.
    contention_table : optional
        Pairwise GMEM bandwidth contention slowdown factors.
    sync_overhead_per_boundary : float
        Extra cycles added per stage boundary for barrier overhead.

    Returns
    -------
    Predicted total cycles for the entire k-iteration loop.
    """
    dispatch_queue = fill_dispatch_queue(k, pipeline)
    t_end, _ = simulate_dispatch_queue(
        dispatch_queue, profiles, dependencies, contention_table
    )
    # Add synchronization overhead: (num_stages - 1) boundaries × overhead
    barrier_overhead = (pipeline.num_stages - 1) * sync_overhead_per_boundary
    return t_end + barrier_overhead


def predict_steady_state_ii(
    pipeline: Pipeline,
    profiles: Dict[str, OpProfile],
    dependencies: Dependencies,
    contention_table: Optional[ContentionTable] = None,
    sync_overhead_per_boundary: float = SYNC_OVERHEAD_PER_STAGE_BOUNDARY,
    warmup_iters: Optional[int] = None,
    measure_iters: int = 32,
) -> float:
    """Predict steady-state initiation interval (II) in cycles per iteration.

    Simulates warmup_iters + measure_iters iterations, then divides the
    time spent in the measure_iters portion by measure_iters to get the
    steady-state II. This avoids including fill/drain phases in the estimate.

    Parameters
    ----------
    pipeline : Pipeline
    profiles : dict
    dependencies : list of (producer, consumer)
    contention_table : optional
    sync_overhead_per_boundary : float
    warmup_iters : optional int
        Number of iterations to skip. If None, automatically set to
        fill_len + 2 to ensure the measurement window is fully in
        steady state. fill_len = sum(shifts) + max(shifts).
    measure_iters : int
        Number of steady-state iterations to measure over.

    Returns
    -------
    Predicted cycles per loop iteration in steady state.
    """
    # Auto-compute warmup from pipeline fill length
    fill_len = sum(pipeline.shifts) + pipeline.max_shift if pipeline.shifts else 0
    if warmup_iters is None:
        warmup_iters = fill_len + 2

    # Add extra iters so the measure window doesn't overlap with drain.
    # Drain length = sum(shifts) for a multi-stage pipeline.
    drain_len = sum(pipeline.shifts) if pipeline.shifts else 0
    k_total = warmup_iters + measure_iters + drain_len
    dispatch_queue = fill_dispatch_queue(k_total, pipeline)
    _, schedule = simulate_dispatch_queue(
        dispatch_queue, profiles, dependencies, contention_table,
    )

    # Identify schedule entries belonging to steady-state iterations
    ss_entries = [
        e for e in schedule
        if warmup_iters <= e.iteration < warmup_iters + measure_iters
    ]

    if not ss_entries:
        # Fallback: use total time / k_total
        t_end = max((e.exec_end for e in schedule), default=0.0)
        return t_end / k_total

    ss_start = min(e.issue_start for e in ss_entries)
    ss_end = max(e.exec_end for e in ss_entries)
    barrier_overhead = (pipeline.num_stages - 1) * sync_overhead_per_boundary

    return (ss_end - ss_start + barrier_overhead) / measure_iters


# ---------------------------------------------------------------------------
# Plan ranking
# ---------------------------------------------------------------------------

def rank_plans(
    plans: List[Pipeline],
    k: int,
    profiles: Dict[str, OpProfile],
    dependencies: Dependencies,
    contention_table: Optional[ContentionTable] = None,
    sync_overhead_per_boundary: float = SYNC_OVERHEAD_PER_STAGE_BOUNDARY,
    top_k: int = 20,
) -> List[Tuple[Pipeline, float]]:
    """Rank pipeline plans by predicted steady-state initiation interval.

    Parameters
    ----------
    plans : list of Pipeline
        Candidate pipeline plans (e.g., from enumerate_pipeline_plans).
    k : int
        Representative iteration count for simulation.
    profiles : dict
        Per-op profiling data.
    dependencies : list of (producer, consumer)
    contention_table : optional
    sync_overhead_per_boundary : float
    top_k : int
        Return only the top-k plans (fastest predicted).

    Returns
    -------
    List of (plan, predicted_II) sorted by predicted_II ascending.
    """
    results: List[Tuple[Pipeline, float]] = []
    for plan in plans:
        predicted_ii = predict_steady_state_ii(
            plan, profiles, dependencies, contention_table,
            sync_overhead_per_boundary
        )
        results.append((plan, predicted_ii))
    results.sort(key=lambda x: x[1])
    return results[:top_k]


# ---------------------------------------------------------------------------
# Profiling plan generation — driven by OpProfilingMeta
# ---------------------------------------------------------------------------

@dataclass
class ProfilingTask:
    """A single profiling measurement to perform on the GPU.

    Generated automatically from OpProfilingMeta; consumed by a profiling
    kernel generator or a manual profiling script.
    """
    kind: str               # "t_issue", "t_exec", or "contention"
    op_name: str            # Primary op being measured
    output_kind: OutputKind # How to force dependency for T_exec
    issue_kind: IssueKind   # How the warp interacts with the op

    # For contention tasks only:
    contention_partner: Optional[str] = None    # Other op in the contention pair
    contention_group: Optional[str] = None      # Which shared resource is contended

    # Setup requirements:
    input_deps: Tuple[str, ...] = ()            # Ops whose outputs must be pre-populated
    needs_smem_setup: bool = False              # Whether SMEM needs pre-population

    def __str__(self) -> str:
        if self.kind == "contention":
            return (f"ProfilingTask(contention: {self.op_name} vs "
                    f"{self.contention_partner}, group={self.contention_group})")
        return f"ProfilingTask({self.kind}: {self.op_name})"


def generate_profiling_plan(
    profiles: Dict[str, OpProfile],
) -> List[ProfilingTask]:
    """Generate an ordered list of profiling tasks from op metadata.

    Given a set of OpProfiles (each carrying OpProfilingMeta), this function
    produces the complete list of measurements needed:

      1. T_issue measurement for every op (always needed).
      2. T_exec measurement for ops where T_tail_hint is not set
         (if T_tail_hint is set, T_tail is known a priori).
      3. Contention measurements for every pair of ops that share at least
         one contention_group.

    The returned list is topologically sorted so that ops with input_deps
    are measured after the ops they depend on (relevant for SMEM setup).

    Parameters
    ----------
    profiles : dict of op_name → OpProfile
        All ops to profile. Each must have a non-None profiling field.

    Returns
    -------
    list of ProfilingTask, in recommended execution order.
    """
    tasks: List[ProfilingTask] = []

    # Topological order: ops with no input_deps first, then their dependents.
    # Simple BFS since the dep graph is small (8-10 ops).
    remaining = dict(profiles)
    resolved: List[str] = []

    while remaining:
        batch = [
            name for name, p in remaining.items()
            if p.profiling is not None
            and all(d in resolved or d not in remaining for d in p.profiling.input_deps)
        ]
        if not batch:
            # No progress — break ties by adding everything left
            batch = list(remaining.keys())
        for name in batch:
            resolved.append(name)
            del remaining[name]

    # Phase 1: T_issue for all ops
    for name in resolved:
        p = profiles[name]
        meta = p.profiling
        if meta is None:
            continue
        tasks.append(ProfilingTask(
            kind="t_issue",
            op_name=name,
            output_kind=meta.output_kind,
            issue_kind=meta.issue_kind,
            input_deps=meta.input_deps,
            needs_smem_setup=meta.needs_smem_setup,
        ))

    # Phase 2: T_exec for ops without T_tail_hint
    for name in resolved:
        p = profiles[name]
        meta = p.profiling
        if meta is None:
            continue
        if meta.T_tail_hint is not None:
            continue  # T_tail is known; skip T_exec measurement
        tasks.append(ProfilingTask(
            kind="t_exec",
            op_name=name,
            output_kind=meta.output_kind,
            issue_kind=meta.issue_kind,
            input_deps=meta.input_deps,
            needs_smem_setup=meta.needs_smem_setup,
        ))

    # Phase 3: Contention measurements for ops sharing contention groups
    contention_pairs = compute_contention_pairs(profiles)
    for (op_a, op_b, group) in contention_pairs:
        meta_a = profiles[op_a].profiling
        meta_b = profiles[op_b].profiling
        if meta_a is None or meta_b is None:
            continue
        # Measure A's tail slowdown when B is concurrent
        tasks.append(ProfilingTask(
            kind="contention",
            op_name=op_a,
            output_kind=meta_a.output_kind,
            issue_kind=meta_a.issue_kind,
            contention_partner=op_b,
            contention_group=group,
            input_deps=meta_a.input_deps + meta_b.input_deps,
            needs_smem_setup=meta_a.needs_smem_setup or meta_b.needs_smem_setup,
        ))
        # Measure B's tail slowdown when A is concurrent
        tasks.append(ProfilingTask(
            kind="contention",
            op_name=op_b,
            output_kind=meta_b.output_kind,
            issue_kind=meta_b.issue_kind,
            contention_partner=op_a,
            contention_group=group,
            input_deps=meta_a.input_deps + meta_b.input_deps,
            needs_smem_setup=meta_a.needs_smem_setup or meta_b.needs_smem_setup,
        ))

    return tasks


def compute_contention_pairs(
    profiles: Dict[str, OpProfile],
) -> List[Tuple[str, str, str]]:
    """Find all pairs of ops that share a contention group.

    Returns a list of (op_a, op_b, group_name) triples, deduplicated
    so that each unordered pair appears exactly once per shared group.

    Parameters
    ----------
    profiles : dict of op_name → OpProfile

    Returns
    -------
    list of (op_a, op_b, group) with op_a < op_b lexicographically.
    """
    # Invert: group → set of op names
    group_to_ops: Dict[str, List[str]] = {}
    for name, p in profiles.items():
        if p.profiling is None:
            continue
        for g in p.profiling.contention_groups:
            group_to_ops.setdefault(g, []).append(name)

    pairs: List[Tuple[str, str, str]] = []
    seen: set = set()
    for group, ops in group_to_ops.items():
        for i, a in enumerate(ops):
            for b in ops[i + 1:]:
                key = (min(a, b), max(a, b), group)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return sorted(pairs)


def print_profiling_plan(tasks: List[ProfilingTask]) -> None:
    """Print a human-readable summary of the profiling plan."""
    t_issue_tasks = [t for t in tasks if t.kind == "t_issue"]
    t_exec_tasks = [t for t in tasks if t.kind == "t_exec"]
    contention_tasks = [t for t in tasks if t.kind == "contention"]

    print(f"Profiling plan: {len(tasks)} total measurements")
    print(f"  {len(t_issue_tasks)} T_issue measurements")
    print(f"  {len(t_exec_tasks)} T_exec measurements")
    print(f"  {len(contention_tasks)} contention measurements "
          f"({len(contention_tasks) // 2} pairs)")
    print()

    print("T_issue measurements:")
    for t in t_issue_tasks:
        print(f"  {t.op_name:32s}  issue={t.issue_kind.value}")

    if t_exec_tasks:
        print("T_exec measurements (no T_tail_hint):")
        for t in t_exec_tasks:
            print(f"  {t.op_name:32s}  output={t.output_kind.value}")

    if contention_tasks:
        print("Contention measurements:")
        printed = set()
        for t in contention_tasks:
            assert t.contention_partner is not None
            pair_key = (min(t.op_name, t.contention_partner),
                        max(t.op_name, t.contention_partner))
            if pair_key not in printed:
                printed.add(pair_key)
                print(f"  {pair_key[0]:32s} vs {pair_key[1]:32s}  "
                      f"group={t.contention_group}")


# ---------------------------------------------------------------------------
# Built-in op profiles for ACC-SpMM (BIT-TCF / MCO format)
# ---------------------------------------------------------------------------
# These are placeholder values based on the design document analysis.
# Replace with measured values from clock() profiling on your target GPU.
#
# Pipe assignments follow the document's op inventory (Section 3).
# T_issue / T_tail are in abstract relative units until measured.

def make_acc_spmm_profiles() -> Dict[str, OpProfile]:
    """
    Return placeholder OpProfile table for the 9-op ACC-SpMM kernel.

    Each op carries OpProfilingMeta describing:
      - output_kind: how to force scoreboard dependency for T_exec
      - issue_kind: how the warp interacts during issue
      - contention_groups: which ops to measure pairwise GMEM contention
      - input_deps: which ops must have produced output before profiling
      - T_tail_hint: known architectural latency (skips T_exec measurement)

    Replace T_issue and T_tail with values measured via clock() instrumentation.
    """
    # Placeholder values for fp32 ACC-SpMM (Tile_K=8, Mma_K=8, float types)
    # Adjusted from fp16 values for smaller tile size and fp32 data width.
    # T_issue: sequential mode with 500-cycle drain between ops
    # T_tail: sandwich method (producer+consumer back-to-back)
    # Replace with measured values from profiling kernel on target GPU.
    return {
        "G2sSparseIndexLoadOp": OpProfile(
            name="G2sSparseIndexLoadOp",
            T_issue=30,
            T_tail=299,
            pipes=Pipe.ASYNC_COPY | Pipe.SMEM_WRITE | Pipe.GMEM_BW,
            bytes_gmem=32,  # Tile_K=8 ints × 4 bytes = 32 bytes
            profiling=OpProfilingMeta(
                output_kind=OutputKind.SMEM_ASYNC,
                issue_kind=IssueKind.ASYNC_CP,
                contention_groups=frozenset({"gmem"}),
            ),
        ),
        "G2rSparseMcoOffLoadOp": OpProfile(
            name="G2rSparseMcoOffLoadOp",
            T_issue=35,
            T_tail=0,  # sandwich ≈ T_issue, likely L1 cache hit
            pipes=Pipe.LDST | Pipe.GMEM_BW,
            bytes_gmem=8,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.REGISTER,
                issue_kind=IssueKind.SYNC_GMEM_LOAD,
                contention_groups=frozenset({"gmem"}),
            ),
        ),
        "G2rSparseMcoMaskLoadOp": OpProfile(
            name="G2rSparseMcoMaskLoadOp",
            T_issue=15,
            T_tail=237,
            pipes=Pipe.LDST | Pipe.GMEM_BW,
            bytes_gmem=16,  # nmask_per_tile=2 × 8 bytes = 16 bytes
            profiling=OpProfilingMeta(
                output_kind=OutputKind.REGISTER,
                issue_kind=IssueKind.SYNC_GMEM_LOAD,
                contention_groups=frozenset({"gmem"}),
            ),
        ),
        "G2sSparseMcoValLoadOp": OpProfile(
            name="G2sSparseMcoValLoadOp",
            T_issue=129,
            T_tail=240,
            pipes=Pipe.ASYNC_COPY | Pipe.SMEM_WRITE | Pipe.CUDA_CORE | Pipe.GMEM_BW,
            bytes_gmem=512,  # 16×8×4 = 512 bytes
            profiling=OpProfilingMeta(
                output_kind=OutputKind.SMEM_ASYNC,
                issue_kind=IssueKind.ASYNC_CP,
                contention_groups=frozenset({"gmem"}),
                input_deps=("G2rSparseMcoOffLoadOp",),
            ),
        ),
        "G2sMatrixBLoadOp": OpProfile(
            name="G2sMatrixBLoadOp",
            T_issue=213,
            T_tail=234,
            pipes=Pipe.ASYNC_COPY | Pipe.SMEM_READ | Pipe.SMEM_WRITE | Pipe.CUDA_CORE | Pipe.GMEM_BW,
            bytes_gmem=2048,  # 8×64×4 = 2048 bytes
            profiling=OpProfilingMeta(
                output_kind=OutputKind.SMEM_ASYNC,
                issue_kind=IssueKind.ASYNC_CP,
                contention_groups=frozenset({"gmem"}),
                input_deps=("G2sSparseIndexLoadOp",),
                needs_smem_setup=True,
            ),
        ),
        "S2sRestoreMatrixAOp": OpProfile(
            name="S2sRestoreMatrixAOp",
            T_issue=800,
            T_tail=0,  # compute-heavy op, warp busy throughout
            pipes=Pipe.CUDA_CORE | Pipe.SMEM_READ | Pipe.SMEM_WRITE,
            bytes_gmem=0,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.SMEM_SYNC,
                issue_kind=IssueKind.COMPUTE_HEAVY,
                input_deps=("G2rSparseMcoMaskLoadOp", "G2sSparseMcoValLoadOp"),
                needs_smem_setup=True,
                T_tail_hint=25.0,  # last st.shared latency
            ),
        ),
        "S2rAValLoadOp": OpProfile(
            name="S2rAValLoadOp",
            T_issue=3,
            T_tail=32,
            pipes=Pipe.SMEM_READ,
            bytes_gmem=0,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.REGISTER,
                issue_kind=IssueKind.SMEM_LOAD,
                input_deps=("S2sRestoreMatrixAOp",),
                needs_smem_setup=True,
                T_tail_hint=25.0,  # ldmatrix SMEM latency
            ),
        ),
        "S2rBValLoadOp": OpProfile(
            name="S2rBValLoadOp",
            T_issue=39,
            T_tail=2,
            pipes=Pipe.SMEM_READ,
            bytes_gmem=0,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.REGISTER,
                issue_kind=IssueKind.SMEM_LOAD,
                input_deps=("G2sMatrixBLoadOp",),
                needs_smem_setup=True,
                T_tail_hint=25.0,  # ldmatrix SMEM latency
            ),
        ),
        "CalculateOp": OpProfile(
            name="CalculateOp",
            T_issue=30,
            T_tail=80,
            pipes=Pipe.TENSOR_CORE,
            bytes_gmem=0,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.REGISTER,
                issue_kind=IssueKind.TENSOR_CORE,
                input_deps=("S2rAValLoadOp", "S2rBValLoadOp"),
                T_tail_hint=32.0,  # m16n8k8 MMA latency
            ),
        ),
    }


# Default dependencies for ACC-SpMM (from design document Section 3)
ACC_SPMM_DEPENDENCIES: Dependencies = [
    ("G2sSparseIndexLoadOp",  "G2sMatrixBLoadOp"),
    ("G2rSparseMcoOffLoadOp", "G2sSparseMcoValLoadOp"),
    ("G2rSparseMcoMaskLoadOp","S2sRestoreMatrixAOp"),
    ("G2sSparseMcoValLoadOp", "S2sRestoreMatrixAOp"),
    ("G2sMatrixBLoadOp",      "S2rBValLoadOp"),
    ("S2sRestoreMatrixAOp",   "S2rAValLoadOp"),
    ("S2rAValLoadOp",         "CalculateOp"),
    ("S2rBValLoadOp",         "CalculateOp"),
]


# ---------------------------------------------------------------------------
# Profile serialization (JSON)
# ---------------------------------------------------------------------------

def _pipe_to_list(pipes: Pipe) -> List[str]:
    """Convert a Pipe bitmask to a sorted list of member names."""
    return sorted(m.name for m in Pipe if m in pipes)


def _list_to_pipe(names: List[str]) -> Pipe:
    """Inverse of _pipe_to_list."""
    result = Pipe(0)
    for n in names:
        result |= Pipe[n]
    return result


def save_profiles(
    profiles: Dict[str, OpProfile],
    path: Union[str, Path],
    *,
    dependencies: Optional[Dependencies] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Save op profiles to a JSON file.

    Parameters
    ----------
    profiles : dict mapping op_name → OpProfile
    path : output file path (typically ``op_profiles.json``)
    dependencies : optional dependency list to embed
    metadata : optional free-form dict (source, timestamp, notes, …)
    """
    ops = {}
    for name, p in profiles.items():
        ops[name] = {
            "T_issue": p.T_issue,
            "T_tail": p.T_tail,
            "pipes": _pipe_to_list(p.pipes),
            "bytes_gmem": p.bytes_gmem,
        }
    doc: dict = {"op_profiles": ops}
    if dependencies is not None:
        doc["dependencies"] = [[a, b] for a, b in dependencies]
    if metadata is not None:
        doc["metadata"] = metadata
    Path(path).write_text(_json.dumps(doc, indent=2) + "\n")


def load_profiles(
    path: Union[str, Path],
    *,
    defaults: Optional[Dict[str, OpProfile]] = None,
) -> Dict[str, OpProfile]:
    """Load op profiles from a JSON file.

    The file must contain an ``"op_profiles"`` key mapping op names to dicts
    with at least ``T_issue``, ``T_tail``, ``pipes``.  If *defaults* is given,
    each loaded profile inherits ``profiling`` (OpProfilingMeta) from the
    matching default, so the caller gets fully-functional OpProfile objects.

    Parameters
    ----------
    path : JSON file written by :func:`save_profiles`.
    defaults : fallback profiles (typically from :func:`make_acc_spmm_profiles`)
        used to fill in ``profiling`` metadata and any ops missing from the file.

    Returns
    -------
    Dict[str, OpProfile]
    """
    doc = _json.loads(Path(path).read_text())
    ops_raw = doc["op_profiles"]

    result: Dict[str, OpProfile] = {}
    for name, d in ops_raw.items():
        profiling_meta = None
        if defaults and name in defaults:
            profiling_meta = defaults[name].profiling
        result[name] = OpProfile(
            name=name,
            T_issue=float(d["T_issue"]),
            T_tail=float(d["T_tail"]),
            pipes=_list_to_pipe(d["pipes"]),
            bytes_gmem=int(d.get("bytes_gmem", 0)),
            profiling=profiling_meta,
        )

    # Fill in any ops present in defaults but missing from the file
    if defaults:
        for name, dp in defaults.items():
            if name not in result:
                result[name] = dp

    return result
