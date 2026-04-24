"""
plan_ranker.py
==============
Bridge between the sparsene pipeline planner (PipelinePlan / NvOpSequence)
and the multi-pipe performance model (perf_model.py).

Provides:
  - nvplan_to_pipeline()     : converts PipelinePlan → Pipeline (perf_model)
  - rank_pipeline_plans()    : ranks a list of PipelinePlan by predicted II
  - print_ranking()          : human-readable ranking table
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sparsene.op_gen.nvir.plan import PipelinePlan
from sparsene.op_gen.nvir.opgraph.graph import OpGraph, construct_graph_from_op_sequence
from sparsene.op_gen.nvir.pipeline.pipeline_planner import enumerate_pipeline_plans, NeighborDependencyValidator
from sparsene.op_gen.nvir.nvop import ForLoopNvOp

from perf_model import (
    ContentionTable,
    Dependencies,
    OpProfile,
    Pipeline,
    predict_steady_state_ii,
    rank_plans,
    ACC_SPMM_DEPENDENCIES,
    make_acc_spmm_profiles,
    load_profiles,
)


# ---------------------------------------------------------------------------
# Conversion: PipelinePlan → perf_model.Pipeline
# ---------------------------------------------------------------------------

def nvplan_to_pipeline(plan: PipelinePlan) -> Pipeline:
    """Convert a PipelinePlan (NvOpSequence-based) to a perf_model Pipeline.

    PipelinePlan.stages is a list of NvOpSequence objects.
    Each NvOpSequence has an .ops attribute listing NvOp objects with .name.

    Parameters
    ----------
    plan : PipelinePlan
        A plan produced by enumerate_pipeline_plans().

    Returns
    -------
    Pipeline
        A perf_model.Pipeline with stages as lists of op name strings.
    """
    stages: List[List[str]] = []
    for seq in plan.stages:
        stages.append([op.name for op in seq.ops])
    return Pipeline(stages=stages, shifts=list(plan.shifts))


def extract_dependencies_from_graph(op_graph: OpGraph) -> Dependencies:
    """Extract same-iteration dependencies from an OpGraph.

    Returns a list of (producer_name, consumer_name) pairs.
    """
    deps: Dependencies = []
    for edge in op_graph.edges:
        deps.append((edge.src.node_id, edge.dst.node_id))
    return deps


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def rank_pipeline_plans(
    plans: List[PipelinePlan],
    profiles: Dict[str, OpProfile],
    dependencies: Dependencies,
    k: int = 10,
    contention_table: Optional[ContentionTable] = None,
    sync_overhead_per_boundary: float = 30.0,
    top_n: int = 20,
) -> List[Tuple[PipelinePlan, float]]:
    """Rank PipelinePlan objects by predicted steady-state initiation interval.

    Parameters
    ----------
    plans : list of PipelinePlan
        Candidate pipeline plans from enumerate_pipeline_plans().
    profiles : dict
        Per-op profiling data (from profiling kernel or make_acc_spmm_profiles()).
    dependencies : list of (producer, consumer)
        Same-iteration data dependencies (from extract_dependencies_from_graph()).
    k : int
        Number of loop iterations used for simulation (representative K-tile count).
    contention_table : optional
        Pairwise GMEM bandwidth contention factors.
    sync_overhead_per_boundary : float
        Cycles added per stage boundary for barrier overhead.
    top_n : int
        Return only the top-N fastest predicted plans.

    Returns
    -------
    list of (PipelinePlan, predicted_II), sorted by predicted_II ascending.
    """
    perf_pipelines = [nvplan_to_pipeline(p) for p in plans]
    ranked_perf = rank_plans(
        plans=perf_pipelines,
        k=k,
        profiles=profiles,
        dependencies=dependencies,
        contention_table=contention_table,
        sync_overhead_per_boundary=sync_overhead_per_boundary,
        top_k=top_n,
    )

    # Map back to original PipelinePlan objects
    # ranked_perf is [(perf_pipeline, predicted_II), ...]
    # We need to align them with the original plans.
    # Build a map from stages-tuple → PipelinePlan for O(1) lookup.
    def plan_key(plan: PipelinePlan) -> tuple:
        return tuple(
            tuple(op.name for op in seq.ops)
            for seq in plan.stages
        ) + tuple(plan.shifts)

    plan_map = {plan_key(p): p for p in plans}

    result: List[Tuple[PipelinePlan, float]] = []
    for perf_pipe, pred_ii in ranked_perf:
        key = tuple(tuple(s) for s in perf_pipe.stages) + tuple(perf_pipe.shifts)
        nv_plan = plan_map.get(key)
        if nv_plan is not None:
            result.append((nv_plan, pred_ii))
    return result


# ---------------------------------------------------------------------------
# Convenience: enumerate + rank in one call
# ---------------------------------------------------------------------------

def enumerate_and_rank(
    for_loop_op: ForLoopNvOp,
    profiles: Optional[Dict[str, OpProfile]] = None,
    dependencies: Optional[Dependencies] = None,
    k: int = 10,
    contention_table: Optional[ContentionTable] = None,
    sync_overhead_per_boundary: float = 30.0,
    top_n: int = 20,
    min_nstages: int = 1,
    max_nstages: int = 3,
    min_shift: int = 1,
    max_shift: int = 3,
) -> List[Tuple[PipelinePlan, float]]:
    """Enumerate all valid pipeline plans and rank them by predicted performance.

    This is the main entry point for profiling-guided plan selection.

    Parameters
    ----------
    for_loop_op : ForLoopNvOp
        The loop op to pipeline (must not already be pipelined).
    profiles : optional dict
        Per-op profiling data. If None, uses placeholder ACC-SpMM profiles.
    dependencies : optional list
        Same-iteration data dependencies. If None, extracts from the op graph.
    k : int
        Representative iteration count for simulation.
    contention_table : optional
        GMEM bandwidth contention factors.
    sync_overhead_per_boundary : float
        Cycles per stage boundary for synchronization overhead.
    top_n : int
        Return only top-N plans.
    min_nstages, max_nstages : int
        Stage count bounds for enumeration.
    min_shift, max_shift : int
        Shift bounds for enumeration.

    Returns
    -------
    list of (PipelinePlan, predicted_II), sorted by predicted_II ascending.
    """
    from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence

    op_graph = construct_graph_from_op_sequence(for_loop_op.body)
    validator = NeighborDependencyValidator(op_graph)

    plans = enumerate_pipeline_plans(
        for_loop_op,
        validator,
        min_nstages=min_nstages,
        max_nstages=max_nstages,
        min_shift=min_shift,
        max_shift=max_shift,
    )
    print(f"Enumerated {len(plans)} valid pipeline plans.")

    if profiles is None:
        defaults = make_acc_spmm_profiles()
        # Try loading measured profiles from file
        _profiles_path = Path(__file__).resolve().parent.parent / "benchmark" / "outputs" / "op_profiles.json"
        if _profiles_path.exists():
            profiles = load_profiles(_profiles_path, defaults=defaults)
            print(f"Loaded profiles from {_profiles_path}")
        else:
            profiles = defaults
            print("Warning: using hardcoded placeholder profiles. "
                  "Run analyze_traces.py to generate op_profiles.json.")

    if dependencies is None:
        dependencies = extract_dependencies_from_graph(op_graph)
        print(f"Extracted {len(dependencies)} dependencies from op graph.")

    ranked = rank_pipeline_plans(
        plans=plans,
        profiles=profiles,
        dependencies=dependencies,
        k=k,
        contention_table=contention_table,
        sync_overhead_per_boundary=sync_overhead_per_boundary,
        top_n=top_n,
    )
    return ranked


# ---------------------------------------------------------------------------
# Pretty-print ranking results
# ---------------------------------------------------------------------------

def print_ranking(
    ranked: List[Tuple[PipelinePlan, float]],
    title: str = "Pipeline Plan Ranking (by predicted initiation interval)",
) -> None:
    """Print ranked plans to stdout in a readable table format.

    Parameters
    ----------
    ranked : list of (PipelinePlan, predicted_II)
    title : str
    """
    print(f"\n{'=' * len(title)}")
    print(title)
    print(f"{'=' * len(title)}")
    print(f"{'Rank':>4}  {'Pred II (cycles)':>18}  Plan")
    print(f"{'----':>4}  {'----------------':>18}  {'----'}")
    for rank, (plan, pred_ii) in enumerate(ranked, start=1):
        print(f"{rank:>4}  {pred_ii:>18.1f}  {plan}")
    print()
