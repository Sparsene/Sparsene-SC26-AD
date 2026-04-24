#!/usr/bin/env python3
"""
simulate_plans.py — Evaluate simulator accuracy on externally-provided plans
=============================================================================

Takes a JSON file containing plans (each described by stages + shifts + actual
kernel time), runs the multi-pipe simulator, and reports ranking accuracy
metrics (Kendall Tau, Spearman Rho, top-k oracle ratio).

Each plan is identified by its (stages, shifts) structure, NOT by any ID.
This makes the file portable across different enumeration systems.

Input file format (JSON):
-------------------------
{
  "plans": [
    {
      "label": "plan_A",                          // optional, for display
      "stages": [
        ["G2sSparseIndexLoadOp", "G2rSparseMcoOffLoadOp"],
        ["G2rSparseMcoMaskLoadOp", "G2sSparseMcoValLoadOp"],
        ["G2sMatrixBLoadOp", "S2sRestoreMatrixAOp"],
        ["S2rAValLoadOp", "S2rBValLoadOp", "CalculateOp"]
      ],
      "shifts": [1, 1, 1],
      "kernel_time_us": 96.31                     // actual measured time
    },
    ...
  ],
  "metadata": {                                   // optional
    "gpu": "H800",
    "matrix": "M=1024 K=1024 N=64 sparsity=0.9"
  }
}

Usage:
------
  # Basic: simulate and evaluate
  python scripts/simulate_plans.py plans_from_colleague.json

  # With custom op profiles
  python scripts/simulate_plans.py plans.json --profiles op_profiles.json

  # Verbose per-plan details
  python scripts/simulate_plans.py plans.json -v

  # Output results to JSON
  python scripts/simulate_plans.py plans.json --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from perf_model import (
    Pipeline,
    predict_steady_state_ii,
    ACC_SPMM_DEPENDENCIES,
    make_acc_spmm_profiles,
    load_profiles,
)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def kendall_tau(pred_rank: list[int], true_rank: list[int]) -> float:
    """Kendall Tau-b rank correlation."""
    n = len(pred_rank)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            p = (pred_rank[i] - pred_rank[j])
            t = (true_rank[i] - true_rank[j])
            if p * t > 0:
                concordant += 1
            elif p * t < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else 0.0


def spearman_rho(pred_rank: list[int], true_rank: list[int]) -> float:
    """Spearman rank correlation coefficient."""
    n = len(pred_rank)
    if n < 2:
        return 1.0
    d_sq = sum((p - t) ** 2 for p, t in zip(pred_rank, true_rank))
    return 1 - 6 * d_sq / (n * (n * n - 1))


def topk_oracle_ratio(sim_times: list[float], real_times: list[float], k: int) -> float:
    """Among the top-k plans by simulation ranking, what fraction of the
    oracle-best performance do we achieve?

    Returns: best_real_time_in_topk / global_best_real_time (closer to 1.0 = better).
    """
    if not sim_times or k <= 0:
        return 0.0
    # Indices sorted by simulation prediction (lower = better)
    sim_order = sorted(range(len(sim_times)), key=lambda i: sim_times[i])
    topk_indices = sim_order[:k]
    best_in_topk = min(real_times[i] for i in topk_indices)
    oracle_best = min(real_times)
    return oracle_best / best_in_topk if best_in_topk > 0 else 0.0


def ndcg_at_k(sim_times: list[float], real_times: list[float], k: int | None = None) -> float:
    """NDCG@k: Normalized Discounted Cumulative Gain.

    Relevance = max(real_times) - real_time (linear, so faster = more relevant).
    Measures how well the simulator ranking places good plans at the top,
    with logarithmic position discount.

    Range: 0.0 (worst) to 1.0 (perfect ranking).
    """
    import math
    n = len(sim_times)
    if n == 0:
        return 0.0
    if k is None:
        k = n

    max_time = max(real_times)
    relevance = [max_time - t for t in real_times]

    # Simulator ranking order
    sim_order = sorted(range(n), key=lambda i: sim_times[i])

    # DCG of simulator ranking
    dcg = sum(relevance[sim_order[i]] / math.log2(i + 2) for i in range(min(k, n)))

    # Ideal DCG (sort by relevance descending)
    ideal_order = sorted(range(n), key=lambda i: relevance[i], reverse=True)
    idcg = sum(relevance[ideal_order[i]] / math.log2(i + 2) for i in range(min(k, n)))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def simulate_plan(stages, shifts, profiles, deps) -> float | None:
    """Run simulator on a single plan, return predicted II."""
    pipeline = Pipeline(stages=stages, shifts=shifts)
    try:
        return predict_steady_state_ii(pipeline, profiles, deps)
    except Exception as e:
        print(f"  WARNING: simulation failed: {e}")
        return None


def evaluate(plans: list[dict], profiles, deps) -> dict:
    """Simulate all plans and compute accuracy metrics."""
    results = []
    for i, plan in enumerate(plans):
        label = plan.get("label", f"plan_{i}")
        stages = plan["stages"]
        shifts = plan["shifts"]
        real_time = plan["kernel_time_us"]

        pred_ii = simulate_plan(stages, shifts, profiles, deps)
        results.append({
            "index": i,
            "label": label,
            "stages": stages,
            "shifts": shifts,
            "kernel_time_us": real_time,
            "predicted_ii": pred_ii,
        })

    # Filter out simulation failures
    valid = [r for r in results if r["predicted_ii"] is not None]
    if len(valid) < 2:
        return {"results": results, "metrics": None, "error": "Too few valid plans"}

    # Compute rankings
    # Real ranking: lower kernel time = better = lower rank number
    real_sorted = sorted(valid, key=lambda r: r["kernel_time_us"])
    real_rank = {r["index"]: rank for rank, r in enumerate(real_sorted)}

    # Predicted ranking: lower predicted II = better
    pred_sorted = sorted(valid, key=lambda r: r["predicted_ii"])
    pred_rank = {r["index"]: rank for rank, r in enumerate(pred_sorted)}

    indices = [r["index"] for r in valid]
    pred_ranks = [pred_rank[i] for i in indices]
    real_ranks = [real_rank[i] for i in indices]

    tau = kendall_tau(pred_ranks, real_ranks)
    rho = spearman_rho(pred_ranks, real_ranks)

    sim_times = [r["predicted_ii"] for r in valid]
    real_times = [r["kernel_time_us"] for r in valid]

    metrics = {
        "n_plans": len(valid),
        "n_failed": len(results) - len(valid),
        "kendall_tau": tau,
        "spearman_rho": rho,
    }

    # Top-k oracle ratios and NDCG
    for k in [1, 3, 5, 10, 20, 50, 100]:
        if k <= len(valid):
            metrics[f"top{k}_oracle_ratio"] = topk_oracle_ratio(sim_times, real_times, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(sim_times, real_times, k)
    metrics["ndcg"] = ndcg_at_k(sim_times, real_times)  # full NDCG

    return {"results": results, "metrics": metrics}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(evaluation: dict, verbose: bool = False):
    results = evaluation["results"]
    metrics = evaluation.get("metrics")

    valid = [r for r in results if r["predicted_ii"] is not None]

    # Per-plan table
    # Sort by predicted II
    valid_sorted = sorted(valid, key=lambda r: r["predicted_ii"])

    # Also compute real rank for display
    real_sorted = sorted(valid, key=lambda r: r["kernel_time_us"])
    real_rank = {r["index"]: i + 1 for i, r in enumerate(real_sorted)}

    print(f"\n{'Sim':>4s}  {'Real':>4s}  {'Label':<20s}  {'Pred II':>9s}  {'Kernel µs':>10s}  ", end="")
    if verbose:
        print("Stages")
    else:
        print()
    print(f"{'----':>4s}  {'----':>4s}  {'----':<20s}  {'---------':>9s}  {'----------':>10s}  ", end="")
    if verbose:
        print("------")
    else:
        print()

    for sim_rank, r in enumerate(valid_sorted, 1):
        rr = real_rank[r["index"]]
        label = r["label"][:20]
        pred = r["predicted_ii"]
        real = r["kernel_time_us"]
        line = f"{sim_rank:4d}  {rr:4d}  {label:<20s}  {pred:9.0f}  {real:10.1f}  "
        if verbose:
            stages_str = " | ".join(",".join(s) for s in r["stages"])
            line += f"[{stages_str}]"
        print(line)

    # Metrics
    if metrics:
        print(f"\n{'='*60}")
        print(f"  Plans evaluated: {metrics['n_plans']}")
        if metrics['n_failed'] > 0:
            print(f"  Simulation failures: {metrics['n_failed']}")
        print(f"  Kendall Tau:   {metrics['kendall_tau']:.4f}")
        print(f"  Spearman Rho:  {metrics['spearman_rho']:.4f}")
        print(f"  NDCG (full):   {metrics.get('ndcg', 0):.4f}")
        print()
        print(f"  {'k':>5s}  {'Oracle%':>8s}  {'NDCG@k':>8s}")
        print(f"  {'---':>5s}  {'-------':>8s}  {'------':>8s}")
        for k in [1, 3, 5, 10, 20, 50, 100]:
            oracle_key = f"top{k}_oracle_ratio"
            ndcg_key = f"ndcg@{k}"
            if oracle_key in metrics:
                oracle_pct = metrics[oracle_key] * 100
                ndcg_val = metrics.get(ndcg_key, 0)
                print(f"  {k:5d}  {oracle_pct:7.1f}%  {ndcg_val:8.4f}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate simulator accuracy on externally-provided plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input file format (JSON):
  {
    "plans": [
      {
        "label": "plan_A",
        "stages": [["OpA","OpB"], ["OpC","OpD"], ...],
        "shifts": [1, 1, ...],
        "kernel_time_us": 96.31
      }, ...
    ]
  }
""")
    p.add_argument("input", help="JSON file with plans (stages + shifts + kernel_time_us)")
    p.add_argument("--profiles", default=None,
                   help="Path to op_profiles.json (default: use built-in placeholder profiles)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show stage details in per-plan table")
    p.add_argument("--output", "-o", default=None,
                   help="Write results + metrics to JSON file")
    args = p.parse_args()

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)
    data = json.loads(input_path.read_text())
    plans = data["plans"]
    print(f"Loaded {len(plans)} plans from {input_path}")
    if "metadata" in data:
        for k, v in data["metadata"].items():
            print(f"  {k}: {v}")

    # Load profiles
    if args.profiles:
        profiles = load_profiles(args.profiles, defaults=make_acc_spmm_profiles())
        print(f"Loaded profiles from {args.profiles}")
    else:
        profiles = make_acc_spmm_profiles()
        print("Using built-in placeholder profiles (pass --profiles for measured values)")

    # Evaluate
    evaluation = evaluate(plans, profiles, ACC_SPMM_DEPENDENCIES)

    # Report
    print_report(evaluation, verbose=args.verbose)

    # Save
    if args.output:
        out = {
            "metrics": evaluation["metrics"],
            "per_plan": [
                {
                    "label": r["label"],
                    "stages": r["stages"],
                    "shifts": r["shifts"],
                    "kernel_time_us": r["kernel_time_us"],
                    "predicted_ii": r["predicted_ii"],
                }
                for r in evaluation["results"]
            ],
        }
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
