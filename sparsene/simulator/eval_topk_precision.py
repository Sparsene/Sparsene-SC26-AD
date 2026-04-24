#!/usr/bin/env python3
import argparse
import json
import random
import statistics
from pathlib import Path


def load_plans(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    plans = data.get("plans")
    if not isinstance(plans, list):
        raise ValueError(f"Invalid file format: {json_path}, missing list field 'plans'.")
    return plans


def topk_precision(subset, k: int, direction: str):
    n = len(subset)
    true_rank = sorted(range(n), key=lambda i: (subset[i]["kernel_time_us"], i))

    if direction == "asc":
        pred_rank = sorted(range(n), key=lambda i: (subset[i]["predicted_ii"], i))
    else:
        pred_rank = sorted(range(n), key=lambda i: (-subset[i]["predicted_ii"], i))

    true_topk = set(true_rank[:k])
    pred_topk = set(pred_rank[:k])
    return len(true_topk & pred_topk) / k


def evaluate(plans, ns, ks, trials: int, seed: int, direction: str):
    rng = random.Random(seed)
    results = {}

    for n in ns:
        if n > len(plans):
            raise ValueError(f"N={n} is larger than total plans={len(plans)}.")

        vals = {k: [] for k in ks}
        for _ in range(trials):
            subset = rng.sample(plans, n)
            for k in ks:
                if k > n:
                    raise ValueError(f"k={k} cannot be larger than N={n}.")
                vals[k].append(topk_precision(subset, k, direction))

        results[n] = {k: statistics.mean(vals[k]) for k in ks}

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate top-k precision between predicted_ii ranking and kernel_time_us ranking."
    )
    parser.add_argument(
        "--json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSON files containing a top-level 'plans' list.",
    )
    parser.add_argument("--ns", type=int, nargs="+", default=[100, 1000, 6000])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 50])
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument(
        "--direction",
        choices=["asc", "desc"],
        default="asc",
        help="How to sort predicted_ii for predicted top-k. 'asc' means smaller is better.",
    )
    parser.add_argument(
        "--merge-inputs",
        action="store_true",
        help="Merge all input files into one plan pool before evaluation.",
    )
    return parser.parse_args()


def print_results(title: str, total_plans: int, ns, ks, results, trials: int, seed: int, direction: str):
    print(f"[{title}] total_plans={total_plans} trials={trials} seed={seed} direction={direction}")
    for n in ns:
        line = ", ".join(f"P@{k}={results[n][k]:.4f}" for k in ks)
        print(f"N={n}: {line}")


def main():
    args = parse_args()

    if args.merge_inputs:
        merged = []
        names = []
        for json_path in args.json:
            plans = load_plans(json_path)
            merged.extend(plans)
            names.append(json_path.name)

        results = evaluate(merged, args.ns, args.ks, args.trials, args.seed, args.direction)
        print_results("merged:" + "+".join(names), len(merged), args.ns, args.ks, results, args.trials, args.seed, args.direction)
        return

    for json_path in args.json:
        plans = load_plans(json_path)
        results = evaluate(plans, args.ns, args.ks, args.trials, args.seed, args.direction)
        print_results(json_path.name, len(plans), args.ns, args.ks, results, args.trials, args.seed, args.direction)


if __name__ == "__main__":
    main()