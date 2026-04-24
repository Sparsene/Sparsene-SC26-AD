#!/usr/bin/env python3
"""
generate_manifest.py — Regenerate manifest from a given plan_searcher.py
========================================================================

Enumerates all pipeline plans using the EXACT same parameters as a specific
plan_searcher.py, producing a manifest.json that maps plan IDs to their
(stages, shifts) descriptions.

This is useful when you receive a result file with plan IDs from a colleague
but need the actual plan structure to feed into the simulator.

Usage:
  # Regenerate manifest from src_fp32/acc/testbed plan_searcher
  python scripts/generate_manifest.py \
      --plan-searcher /path/to/src_fp32/acc/testbed/scripts/plan_searcher.py \
      --min-stages 2 --max-stages 3 \
      --min-ops-per-stage 2 --max-ops-per-stage 4 \
      --max-shift 3 \
      -o colleague_manifest.json

  # Then convert a result.json to simulate_plans.py input format
  python scripts/generate_manifest.py \
      --plan-searcher /path/to/plan_searcher.py \
      --result-json /path/to/result.json \
      -o plans_for_simulation.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_acc_function(plan_searcher_path: str):
    """Dynamically import acc() from an arbitrary plan_searcher.py."""
    spec = importlib.util.spec_from_file_location("plan_searcher_ext", plan_searcher_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.acc


def enumerate_plans(acc_fn, args):
    """Enumerate plans using the same pipeline planner."""
    from sparsene.op_gen.nvir.pipeline.pipeline_planner import (
        enumerate_pipeline_plans,
        NeighborDependencyValidator,
    )
    from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence

    main_loop_op, program = acc_fn()
    op_graph = construct_graph_from_op_sequence(main_loop_op.body)
    validator = NeighborDependencyValidator(op_graph)

    plans = enumerate_pipeline_plans(
        main_loop_op,
        validator,
        min_nstages=args.min_stages,
        max_nstages=args.max_stages,
        min_ops_per_stage=args.min_ops_per_stage,
        max_ops_per_stage=args.max_ops_per_stage,
        min_shift=1,
        max_shift=args.max_shift,
    )
    return plans


def build_manifest(plans) -> list[dict]:
    """Convert enumerated plans to manifest entries."""
    manifest = []
    for idx, plan in enumerate(plans):
        manifest.append({
            "id": idx,
            "stages": [[op.name for op in stage.ops] for stage in plan.stages],
            "shifts": list(plan.shifts),
        })
    return manifest


def convert_result_to_simulation_input(manifest: list[dict], result_json: dict) -> dict:
    """Convert a result.json (with plan IDs + kernel times) into the
    simulate_plans.py input format (with stages + shifts + kernel_time_us).

    Handles multiple result.json formats:
      - {"plans": [{"id": N, "kernel_time_us": T}, ...]}
      - [{"id": N, "kernel_time_us": T}, ...]
      - {"results": {"<id>": T, ...}}
      - {"top10": [{"id": N, "time_us": T}, ...], ...}
    """
    id_to_entry = {e["id"]: e for e in manifest}

    # Try to extract (plan_id, kernel_time) pairs from various formats
    pairs = []

    if isinstance(result_json, list):
        # Format: [{"id": N, "kernel_time_us": T}, ...]
        for item in result_json:
            pid = item.get("id", item.get("plan_id"))
            t = item.get("kernel_time_us", item.get("time_us", item.get("kernel_time", item.get("time"))))
            if pid is not None and t is not None:
                t = float(t)
                # Auto-detect unit: if < 1, likely ms → convert to µs
                if t < 10:
                    t = t * 1000  # ms → µs
                pairs.append((int(pid), t))

    elif isinstance(result_json, dict):
        # Format: {"plans": [...]}
        if "plans" in result_json:
            for item in result_json["plans"]:
                pid = item.get("id", item.get("plan_id"))
                t = item.get("kernel_time_us", item.get("time_us", item.get("time")))
                if pid is not None and t is not None:
                    pairs.append((int(pid), float(t)))

        # Format: {"results": {"123": 96.3, ...}} or {"results": [{"id":..., "time_us":...}]}
        elif "results" in result_json:
            r = result_json["results"]
            if isinstance(r, dict):
                for pid_str, t in r.items():
                    pairs.append((int(pid_str), float(t)))
            elif isinstance(r, list):
                for item in r:
                    pid = item.get("id", item.get("plan_id"))
                    t = item.get("kernel_time_us", item.get("time_us", item.get("time")))
                    if pid is not None and t is not None:
                        pairs.append((int(pid), float(t)))

        # Format: {"top10": [{"id": N, "time_us": T}], ...}  (best_plan.json style)
        elif "top10" in result_json:
            for item in result_json["top10"]:
                pairs.append((int(item["id"]), float(item["time_us"])))

        # Format: flat {"0": 96.3, "1": 97.0, ...}
        else:
            for key, val in result_json.items():
                try:
                    pairs.append((int(key), float(val)))
                except (ValueError, TypeError):
                    pass

    if not pairs:
        print("ERROR: Could not parse any (plan_id, kernel_time) from result.json")
        print(f"  Keys found: {list(result_json.keys()) if isinstance(result_json, dict) else 'list'}")
        sys.exit(1)

    # Build simulation input
    sim_plans = []
    missing = []
    for pid, t in pairs:
        if pid in id_to_entry:
            entry = id_to_entry[pid]
            sim_plans.append({
                "label": f"plan_{pid}",
                "stages": entry["stages"],
                "shifts": entry["shifts"],
                "kernel_time_us": t,
            })
        else:
            missing.append(pid)

    if missing:
        print(f"WARNING: {len(missing)} plan IDs not found in manifest: {missing[:10]}{'...' if len(missing)>10 else ''}")

    return {
        "plans": sim_plans,
        "metadata": {
            "source": "generated from result.json + regenerated manifest",
            "n_plans": len(sim_plans),
            "n_missing": len(missing),
        },
    }


def main():
    p = argparse.ArgumentParser(
        description="Regenerate manifest from plan_searcher.py and optionally convert result.json")
    p.add_argument("--plan-searcher", required=True,
                   help="Path to the plan_searcher.py used to generate the plans")
    p.add_argument("--min-stages", type=int, default=2)
    p.add_argument("--max-stages", type=int, default=3)
    p.add_argument("--min-ops-per-stage", type=int, default=2)
    p.add_argument("--max-ops-per-stage", type=int, default=4)
    p.add_argument("--max-shift", type=int, default=3)
    p.add_argument("--result-json", default=None,
                   help="Optional: result.json with plan IDs + kernel times. "
                        "If given, output is in simulate_plans.py input format.")
    p.add_argument("-o", "--output", required=True, help="Output JSON file")
    args = p.parse_args()

    # Load acc() from the specified plan_searcher
    print(f"Loading acc() from {args.plan_searcher}")
    acc_fn = load_acc_function(args.plan_searcher)

    # Enumerate plans
    print(f"Enumerating plans (stages={args.min_stages}-{args.max_stages}, "
          f"ops/stage={args.min_ops_per_stage}-{args.max_ops_per_stage}, "
          f"max_shift={args.max_shift})...")
    plans = enumerate_plans(acc_fn, args)
    print(f"  Found {len(plans)} plans")

    manifest = build_manifest(plans)

    if args.result_json:
        # Convert result.json → simulate_plans.py input
        print(f"Loading result.json from {args.result_json}")
        result_data = json.loads(Path(args.result_json).read_text())
        output = convert_result_to_simulation_input(manifest, result_data)
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(f"Wrote {len(output['plans'])} plans to {args.output}")
        print(f"  (simulate_plans.py input format — ready to use)")
    else:
        # Just output the manifest
        Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote manifest ({len(manifest)} plans) to {args.output}")

    # Print sample
    print(f"\nSample (first 3 plans):")
    for e in manifest[:3]:
        stages_str = " | ".join(",".join(s) for s in e["stages"])
        print(f"  {e['id']:4d}: [{stages_str}]  shifts={e['shifts']}")


if __name__ == "__main__":
    main()
