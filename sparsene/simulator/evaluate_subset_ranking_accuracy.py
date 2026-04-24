#!/usr/bin/env python3
"""Evaluate simulator ranking accuracy on uniformly sampled subsets.

Sampling strategy:
1) Partition full plans by equal-width bins on kernel_time_ms.
2) Perform round-robin sampling across non-empty bins so sampled points
   cover the whole time range more evenly.

Metrics per subset:
- Precision@1/5/10/50 (top-k overlap between predicted and true ranking)
- Kendall Tau (for permutations, inversion-count implementation)
- Spearman Rho (rank correlation for permutations)
- NDCG@50 and NDCG@All
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ranking accuracy on uniformly sampled plan subsets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results.json"),
        help="Path to combined_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output CSV/JSON files",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="1000,2000,3000,4000,5000,6000",
        help="Comma-separated subset sizes",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of random samplings per subset size",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=100,
        help="Equal-width bin count for uniform-range sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260401,
        help="Random seed",
    )
    parser.add_argument(
        "--true-key",
        type=str,
        default="kernel_time_ms",
        help="Ground-truth ranking key (smaller is better)",
    )
    parser.add_argument(
        "--pred-key",
        type=str,
        default="predicted_ii",
        help="Predicted ranking key (smaller is better)",
    )
    return parser.parse_args()


def load_plans(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    plans = data.get("plans", [])
    if not isinstance(plans, list) or not plans:
        raise ValueError("Input JSON does not contain a valid non-empty 'plans' list.")
    return plans


def parse_sample_sizes(text: str) -> List[int]:
    sizes: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        sizes.append(int(token))
    if not sizes:
        raise ValueError("No valid sample size provided.")
    return sorted(set(sizes))


def get_plan_uid(plan: dict, fallback_idx: int) -> int:
    pid = plan.get("id")
    if isinstance(pid, int):
        return pid
    return fallback_idx


def build_equal_width_bins(
    plans: Sequence[dict], time_key: str, num_bins: int
) -> List[List[int]]:
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")

    values = [float(p[time_key]) for p in plans]
    t_min, t_max = min(values), max(values)

    bins: List[List[int]] = [[] for _ in range(num_bins)]
    if t_max == t_min:
        bins[0] = list(range(len(plans)))
        return bins

    width = (t_max - t_min) / num_bins
    for idx, val in enumerate(values):
        if val >= t_max:
            b = num_bins - 1
        else:
            b = int((val - t_min) / width)
            if b < 0:
                b = 0
            elif b >= num_bins:
                b = num_bins - 1
        bins[b].append(idx)
    return bins


def sample_uniform_over_time_bins(
    bins: Sequence[Sequence[int]],
    sample_size: int,
    rng: random.Random,
) -> List[int]:
    total = sum(len(b) for b in bins)
    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds total plans={total}")

    working = [list(bucket) for bucket in bins]
    for bucket in working:
        rng.shuffle(bucket)

    active = [i for i, bucket in enumerate(working) if bucket]
    selected: List[int] = []

    while len(selected) < sample_size:
        if not active:
            raise RuntimeError("Not enough items to complete sampling.")

        rng.shuffle(active)
        next_active: List[int] = []

        for b in active:
            if not working[b]:
                continue
            selected.append(working[b].pop())
            if working[b]:
                next_active.append(b)
            if len(selected) >= sample_size:
                break

        if len(selected) < sample_size:
            active = next_active

    return selected


def precision_at_k(order_true: Sequence[int], order_pred: Sequence[int], k: int) -> float:
    if k <= 0:
        return float("nan")
    k = min(k, len(order_true), len(order_pred))
    if k == 0:
        return float("nan")
    true_topk = set(order_true[:k])
    hits = sum(1 for pid in order_pred[:k] if pid in true_topk)
    return hits / k


def _merge_count(arr: List[int]) -> int:
    if len(arr) <= 1:
        return 0
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    inv = _merge_count(left) + _merge_count(right)

    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
            inv += len(left) - i
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
    return inv


def kendall_tau_permutation(order_true: Sequence[int], order_pred: Sequence[int]) -> float:
    n = len(order_true)
    if n < 2:
        return float("nan")
    true_rank = {pid: r for r, pid in enumerate(order_true)}
    perm = [true_rank[pid] for pid in order_pred]
    inv = _merge_count(perm)
    total_pairs = n * (n - 1) / 2
    return 1.0 - (2.0 * inv) / total_pairs


def spearman_rho_permutation(order_true: Sequence[int], order_pred: Sequence[int]) -> float:
    n = len(order_true)
    if n < 2:
        return float("nan")
    true_rank = {pid: r for r, pid in enumerate(order_true)}
    pred_rank = {pid: r for r, pid in enumerate(order_pred)}
    d2 = 0
    for pid in order_true:
        d = true_rank[pid] - pred_rank[pid]
        d2 += d * d
    denom = n * (n * n - 1)
    return 1.0 - (6.0 * d2) / denom


def ndcg(order_true: Sequence[int], order_pred: Sequence[int], k: int | None = None) -> float:
    n = len(order_true)
    if n == 0:
        return float("nan")

    if k is None:
        k = n
    k = max(1, min(k, n))

    true_rank = {pid: r for r, pid in enumerate(order_true)}

    def gain_from_true_rank(rank: int) -> float:
        return float(n - rank)

    dcg = 0.0
    for i, pid in enumerate(order_pred[:k]):
        rel = gain_from_true_rank(true_rank[pid])
        dcg += rel / math.log2(i + 2.0)

    idcg = 0.0
    for i in range(k):
        rel = gain_from_true_rank(i)
        idcg += rel / math.log2(i + 2.0)

    if idcg == 0.0:
        return float("nan")
    return dcg / idcg


def summarize(records: Sequence[Dict[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(r[key]) for r in records]
        out[f"{key}_mean"] = statistics.fmean(values)
        out[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return out


def evaluate(
    plans: Sequence[dict],
    sample_sizes: Sequence[int],
    repeats: int,
    bins: Sequence[Sequence[int]],
    true_key: str,
    pred_key: str,
    seed: int,
) -> Dict[str, object]:
    metric_keys = [
        "p_at_1",
        "p_at_5",
        "p_at_10",
        "p_at_50",
        "kendall_tau",
        "spearman_rho",
        "ndcg_at_50",
        "ndcg_all",
    ]
    details: List[Dict[str, float]] = []
    summary: List[Dict[str, float]] = []

    for n in sample_sizes:
        size_records: List[Dict[str, float]] = []
        for rep in range(repeats):
            run_seed = seed + n * 1000 + rep
            rng = random.Random(run_seed)
            subset_idx = sample_uniform_over_time_bins(bins, n, rng)

            order_true = sorted(
                subset_idx,
                key=lambda i: (float(plans[i][true_key]), get_plan_uid(plans[i], i)),
            )
            order_pred = sorted(
                subset_idx,
                key=lambda i: (float(plans[i][pred_key]), get_plan_uid(plans[i], i)),
            )

            rec: Dict[str, float] = {
                "sample_size": float(n),
                "repeat": float(rep),
                "seed": float(run_seed),
                "p_at_1": precision_at_k(order_true, order_pred, 1),
                "p_at_5": precision_at_k(order_true, order_pred, 5),
                "p_at_10": precision_at_k(order_true, order_pred, 10),
                "p_at_50": precision_at_k(order_true, order_pred, 50),
                "kendall_tau": kendall_tau_permutation(order_true, order_pred),
                "spearman_rho": spearman_rho_permutation(order_true, order_pred),
                "ndcg_at_50": ndcg(order_true, order_pred, k=50),
                "ndcg_all": ndcg(order_true, order_pred, k=None),
            }
            details.append(rec)
            size_records.append(rec)

        s = summarize(size_records, metric_keys)
        s.update(
            {
                "sample_size": float(n),
                "repeats": float(repeats),
            }
        )
        summary.append(s)

    return {
        "summary": summary,
        "details": details,
    }


def write_csv(path: Path, records: Sequence[Dict[str, float]]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    plans = load_plans(args.input)

    sample_sizes = parse_sample_sizes(args.sample_sizes)
    n_total = len(plans)
    for n in sample_sizes:
        if n <= 0 or n > n_total:
            raise ValueError(f"Invalid sample size {n}, must be in [1, {n_total}]")

    for key in (args.true_key, args.pred_key):
        if key not in plans[0]:
            raise ValueError(f"Key '{key}' not found in plan entries")

    bins = build_equal_width_bins(plans, args.true_key, args.num_bins)
    n_nonempty_bins = sum(1 for b in bins if b)

    results = evaluate(
        plans=plans,
        sample_sizes=sample_sizes,
        repeats=args.repeats,
        bins=bins,
        true_key=args.true_key,
        pred_key=args.pred_key,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_csv = args.output_dir / "subset_accuracy_details.csv"
    summary_csv = args.output_dir / "subset_accuracy_summary.csv"
    summary_json = args.output_dir / "subset_accuracy_summary.json"

    # write_csv(details_csv, results["details"])
    # write_csv(summary_csv, results["summary"])

    payload = {
        "config": {
            "input": str(args.input),
            "n_total_plans": n_total,
            "sample_sizes": sample_sizes,
            "repeats": args.repeats,
            "num_bins": args.num_bins,
            "nonempty_bins": n_nonempty_bins,
            "true_key": args.true_key,
            "pred_key": args.pred_key,
            "seed": args.seed,
        },
        "summary": results["summary"],
    }
    # with summary_json.open("w", encoding="utf-8") as f:
    #     json.dump(payload, f, indent=2)

    print("=== Subset Ranking Accuracy Evaluation ===")
    print(f"Input plans: {n_total}")
    print(f"Sampling bins: {args.num_bins} (non-empty: {n_nonempty_bins})")
    print(f"Sample sizes: {sample_sizes}, repeats: {args.repeats}")
    print(f"Details CSV: {details_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("")
    print("sample_size | P@1 | P@5 | P@10 | P@50 | tau | rho | NDCG@50 | NDCG@All")
    for row in results["summary"]:
        print(
            f"{int(row['sample_size']):10d} | "
            f"{row['p_at_1_mean']:.4f} | "
            f"{row['p_at_5_mean']:.4f} | "
            f"{row['p_at_10_mean']:.4f} | "
            f"{row['p_at_50_mean']:.4f} | "
            f"{row['kendall_tau_mean']:.4f} | "
            f"{row['spearman_rho_mean']:.4f} | "
            f"{row['ndcg_at_50_mean']:.4f} | "
            f"{row['ndcg_all_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
