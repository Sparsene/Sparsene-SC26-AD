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
    ASYNC_COPY  = auto()  # cp.async engine (background GMEM->SMEM transfer)
    LDST        = auto()  # Synchronous load/store unit (ld.global, st.global)
    TENSOR_CORE = auto()  # MMA unit (mma.sync / HMMA instructions)
    CUDA_CORE   = auto()  # INT/FP ALU (arithmetic, address computation, popcll)
    SMEM_READ   = auto()  # Shared memory read (ldmatrix, ld.shared)
    SMEM_WRITE  = auto()  # Shared memory write (st.shared, cp.async landing)
    GMEM_BW     = auto()  # Virtual: GMEM bandwidth (for contention tracking only)

    # Convenience alias
    @classmethod
    @property
    def SMEM(cls) -> "Pipe":
        return cls.SMEM_READ | cls.SMEM_WRITE

# Pipes that represent real hardware units (used for conflict detection)
HARDWARE_PIPES = (Pipe.ASYNC_COPY | Pipe.LDST | Pipe.TENSOR_CORE | Pipe.CUDA_CORE
                  | Pipe.SMEM_READ | Pipe.SMEM_WRITE)

# ---------------------------------------------------------------------------
# Profiling metadata
# ---------------------------------------------------------------------------

class OutputKind(Enum):
    REGISTER   = "register"
    SMEM_ASYNC = "smem_async"
    SMEM_SYNC  = "smem_sync"
    NONE       = "none"


class IssueKind(Enum):
    ASYNC_CP       = "async_cp"
    SYNC_GMEM_LOAD = "sync_gmem_load"
    COMPUTE_HEAVY  = "compute_heavy"
    SMEM_LOAD      = "smem_load"
    TENSOR_CORE    = "tensor_core"


@dataclass
class OpProfilingMeta:
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
    name: str
    T_issue: float
    T_tail: float
    pipes: Pipe
    bytes_gmem: int = 0
    profiling: Optional[OpProfilingMeta] = None

    @property
    def T_exec(self) -> float:
        return self.T_issue + self.T_tail

    @property
    def uses_gmem_bw(self) -> bool:
        return bool(self.pipes & Pipe.GMEM_BW)

    @property
    def hardware_pipes(self) -> Pipe:
        return self.pipes & HARDWARE_PIPES


# ---------------------------------------------------------------------------
# Contention table
# ---------------------------------------------------------------------------

ContentionTable = Dict[str, Dict[str, float]]


def no_contention_table() -> ContentionTable:
    return {}


# ---------------------------------------------------------------------------
# Schedule entry (simulation output)
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    op_name: str
    iteration: int
    issue_start: float
    issue_end: float
    exec_start: float
    exec_end: float


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

Dependencies = List[Tuple[str, str]]


def build_consumer_to_producers(dependencies: Dependencies) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for producer, consumer in dependencies:
        result.setdefault(consumer, []).append(producer)
        result.setdefault(producer, [])
    return result


# ---------------------------------------------------------------------------
# Core simulation: dispatch queue -> schedule
# ---------------------------------------------------------------------------

def simulate_dispatch_queue(
    dispatch_queue: List[Tuple[str, int]],
    profiles: Dict[str, OpProfile],
    dependencies: Dependencies,
    contention_table: Optional[ContentionTable] = None,
) -> Tuple[float, List[ScheduleEntry]]:
    if contention_table is None:
        contention_table = {}

    consumer_to_producers = build_consumer_to_producers(dependencies)

    warp_issue_time: float = 0.0
    pipe_free: Dict[Pipe, float] = {p: 0.0 for p in Pipe}
    op_exec_end: Dict[Tuple[str, int], float] = {}
    gmem_in_flight: List[Tuple[float, str]] = []

    schedule: List[ScheduleEntry] = []

    for op_name, iteration in dispatch_queue:
        p = profiles[op_name]

        issue_start = warp_issue_time
        for producer in consumer_to_producers.get(op_name, []):
            key = (producer, iteration)
            if key in op_exec_end:
                issue_start = max(issue_start, op_exec_end[key])

        issue_end = issue_start + p.T_issue

        exec_start = issue_start
        if p.pipes & Pipe.TENSOR_CORE:
            exec_start = max(exec_start, pipe_free.get(Pipe.TENSOR_CORE, 0.0))

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

        warp_issue_time = issue_end

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
        return self.max_shift + 1


def fill_dispatch_queue(
    k: int,
    pipeline: Pipeline,
) -> List[Tuple[str, int]]:
    dispatch_queue: List[Tuple[str, int]] = []
    fill_len = sum(pipeline.shifts) + pipeline.max_shift
    nbuf = pipeline.nbuf

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
        stages = pipeline.stages
        shifts = pipeline.shifts
        for i in range(k + sum(shifts)):
            for stage_idx, stage in enumerate(stages):
                shift_val = sum(shifts[:stage_idx])
                if 0 <= i - shift_val < k:
                    for op in stage:
                        dispatch_queue.append((op, i - shift_val))

    def fill_dispatch() -> None:
        for i in range(len(pipeline.stages)):
            nsteps = pipeline.shifts[i] if i < len(pipeline.shifts) else pipeline.max_shift
            for _ in range(nsteps):
                for stage in pipeline.stages[: i + 1]:
                    for op in stage:
                        dispatch_queue.append((op, _next_id(op)))

    def loop_step_dispatch() -> None:
        for _ in range(nbuf):
            for stage in pipeline.stages:
                for op in stage:
                    dispatch_queue.append((op, _next_id(op)))

    def remainder_dispatch(i: int) -> None:
        remain = k - i
        if remain <= 0:
            return
        for _ in range(remain):
            for stage in pipeline.stages:
                for op in stage:
                    dispatch_queue.append((op, _next_id(op)))

    def empty_dispatch(i: int) -> None:
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

SYNC_OVERHEAD_PER_STAGE_BOUNDARY: float = 0.0


def predict_pipeline_time(
    pipeline: Pipeline,
    k: int,
    profiles: Dict[str, OpProfile],
    dependencies: Dependencies,
    contention_table: Optional[ContentionTable] = None,
    sync_overhead_per_boundary: float = SYNC_OVERHEAD_PER_STAGE_BOUNDARY,
) -> float:
    dispatch_queue = fill_dispatch_queue(k, pipeline)
    t_end, _ = simulate_dispatch_queue(
        dispatch_queue, profiles, dependencies, contention_table
    )
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
    fill_len = sum(pipeline.shifts) + pipeline.max_shift if pipeline.shifts else 0
    if warmup_iters is None:
        warmup_iters = fill_len + 2

    drain_len = sum(pipeline.shifts) if pipeline.shifts else 0
    k_total = warmup_iters + measure_iters + drain_len
    dispatch_queue = fill_dispatch_queue(k_total, pipeline)
    _, schedule = simulate_dispatch_queue(
        dispatch_queue, profiles, dependencies, contention_table,
    )

    ss_entries = [
        e for e in schedule
        if warmup_iters <= e.iteration < warmup_iters + measure_iters
    ]

    if not ss_entries:
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
# Profiling plan generation
# ---------------------------------------------------------------------------

@dataclass
class ProfilingTask:
    kind: str
    op_name: str
    output_kind: OutputKind
    issue_kind: IssueKind
    contention_partner: Optional[str] = None
    contention_group: Optional[str] = None
    input_deps: Tuple[str, ...] = ()
    needs_smem_setup: bool = False

    def __str__(self) -> str:
        if self.kind == "contention":
            return (f"ProfilingTask(contention: {self.op_name} vs "
                    f"{self.contention_partner}, group={self.contention_group})")
        return f"ProfilingTask({self.kind}: {self.op_name})"


def generate_profiling_plan(
    profiles: Dict[str, OpProfile],
) -> List[ProfilingTask]:
    tasks: List[ProfilingTask] = []

    remaining = dict(profiles)
    resolved: List[str] = []

    while remaining:
        batch = [
            name for name, p in remaining.items()
            if p.profiling is not None
            and all(d in resolved or d not in remaining for d in p.profiling.input_deps)
        ]
        if not batch:
            batch = list(remaining.keys())
        for name in batch:
            resolved.append(name)
            del remaining[name]

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

    for name in resolved:
        p = profiles[name]
        meta = p.profiling
        if meta is None:
            continue
        if meta.T_tail_hint is not None:
            continue
        tasks.append(ProfilingTask(
            kind="t_exec",
            op_name=name,
            output_kind=meta.output_kind,
            issue_kind=meta.issue_kind,
            input_deps=meta.input_deps,
            needs_smem_setup=meta.needs_smem_setup,
        ))

    contention_pairs = compute_contention_pairs(profiles)
    for (op_a, op_b, group) in contention_pairs:
        meta_a = profiles[op_a].profiling
        meta_b = profiles[op_b].profiling
        if meta_a is None or meta_b is None:
            continue
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
# Built-in op profiles for BitBSR-SpMM (BIT-TCF / MCO format)
# ---------------------------------------------------------------------------

def make_bitbsr_spmm_profiles() -> Dict[str, OpProfile]:
    """
    Return placeholder OpProfile table for the 9-op BitBSR-SpMM kernel.

    Key difference from ACC: Op 0 is G2rSparseIndexLoadOp (ld.global -> register)
    instead of G2sSparseIndexLoadOp (cp.async -> smem). This changes its pipes
    from ASYNC_COPY | SMEM_WRITE | GMEM_BW to LDST | GMEM_BW.
    """
    return {
        "G2rSparseIndexLoadOp": OpProfile(
            name="G2rSparseIndexLoadOp",
            T_issue=30,
            T_tail=299,
            pipes=Pipe.LDST | Pipe.GMEM_BW,
            bytes_gmem=4,  # 1 int = 4 bytes (single ld.global)
            profiling=OpProfilingMeta(
                output_kind=OutputKind.REGISTER,
                issue_kind=IssueKind.SYNC_GMEM_LOAD,
                contention_groups=frozenset({"gmem"}),
            ),
        ),
        "G2rSparseMcoOffLoadOp": OpProfile(
            name="G2rSparseMcoOffLoadOp",
            T_issue=35,
            T_tail=0,
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
            bytes_gmem=16,
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
            bytes_gmem=512,
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
            pipes=Pipe.ASYNC_COPY | Pipe.SMEM_WRITE | Pipe.CUDA_CORE | Pipe.GMEM_BW,
            bytes_gmem=2048,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.SMEM_ASYNC,
                issue_kind=IssueKind.ASYNC_CP,
                contention_groups=frozenset({"gmem"}),
                input_deps=("G2rSparseIndexLoadOp",),
                needs_smem_setup=True,
            ),
        ),
        "S2sRestoreMatrixAOp": OpProfile(
            name="S2sRestoreMatrixAOp",
            T_issue=800,
            T_tail=0,
            pipes=Pipe.CUDA_CORE | Pipe.SMEM_READ | Pipe.SMEM_WRITE,
            bytes_gmem=0,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.SMEM_SYNC,
                issue_kind=IssueKind.COMPUTE_HEAVY,
                input_deps=("G2rSparseMcoMaskLoadOp", "G2sSparseMcoValLoadOp"),
                needs_smem_setup=True,
                T_tail_hint=25.0,
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
                T_tail_hint=25.0,
            ),
        ),
        "S2rBValLoadOp": OpProfile(
            name="S2rBValLoadOp",
            T_issue=39,
            T_tail=2,
            pipes=Pipe.SMEM_READ | Pipe.CUDA_CORE,
            bytes_gmem=0,
            profiling=OpProfilingMeta(
                output_kind=OutputKind.REGISTER,
                issue_kind=IssueKind.SMEM_LOAD,
                input_deps=("G2sMatrixBLoadOp",),
                needs_smem_setup=True,
                T_tail_hint=25.0,
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
                T_tail_hint=32.0,
            ),
        ),
    }


# Default dependencies for BitBSR-SpMM
BITBSR_SPMM_DEPENDENCIES: Dependencies = [
    ("G2rSparseIndexLoadOp",  "G2sMatrixBLoadOp"),
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
    return sorted(m.name for m in Pipe if m in pipes)


def _list_to_pipe(names: List[str]) -> Pipe:
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

    if defaults:
        for name, dp in defaults.items():
            if name not in result:
                result[name] = dp

    return result
