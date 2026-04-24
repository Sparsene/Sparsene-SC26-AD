import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

#! FP32
DEVICE_TO_FILE_FP32 = {
    "A100": "a100_fp32_merged.json",
    "H100": "h800_fp32_merged.json",
    "4090": "4090_fp32_merged.json",
}


# BASELINES = ["FlashSparse", "SparseTIR", "sputnik", "DTC-SPMM", "ACC-SPMM", "cusparse"]
#! FP32
BASELINES = ["cusparse", "sputnik", "DTC-SPMM", "ACC-SPMM", "FlashSparse", "SparseTIR"]
#! FP16
# BASELINES = ["cusparse", "FlashSparse", "SparseTIR"]
N_ROWS = ["128", "256", "512"]


def load_json(path: Path):
    if not path.exists():
        print(f"Error: File not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def avg_sparsene_vs_baseline(data, n_value: str, baseline: str):
    values = []
    for matrix_name, by_n in data.items():
        vals = by_n.get(n_value)
        if not vals:
            continue
        s = vals.get("sparsene")
        b = vals.get(baseline)
        if s is None or b is None or s <= 0 or b <= 0:
            continue
        # Larger is better: how many times baseline is slower than sparsene.
        values.append(b / s)

    if not values:
        return None
    return float(fmean(values))


def collect_table(result_dir: Path):
    table = []
    for device, file_name in DEVICE_TO_FILE_FP32.items():
        data = load_json(result_dir / file_name)

        row_values = {}
        for n_value in N_ROWS:
            row = {}
            for baseline in BASELINES:
                row[baseline] = avg_sparsene_vs_baseline(data, n_value, baseline)
            row_values[n_value] = row

        avg_row = {}
        for baseline in BASELINES:
            vals = [row_values[n][baseline] for n in N_ROWS if row_values[n][baseline] is not None]
            avg_row[baseline] = float(fmean(vals)) if vals else None

        for n_value in N_ROWS:
            table.append({"GPU": device, "N": n_value, **row_values[n_value]})
        table.append({"GPU": device, "N": "average", **avg_row})

    return table


def fmt(v):
    if v is None:
        return "N/A"
    return f"{v:.2f}"


def write_csv(table, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GPU", "N", *BASELINES])
        for row in table:
            writer.writerow([row["GPU"], row["N"], *[fmt(row[b]) for b in BASELINES]])
    print(f"Saved CSV: {path}")


def write_latex(table, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("\\begin{tabular}{ll" + "c" * len(BASELINES) + "}")
    lines.append("\\toprule")
    lines.append("GPU & N & " + " & ".join(BASELINES) + " \\\\")
    lines.append("\\midrule")

    current_gpu = None
    for row in table:
        gpu = row["GPU"]
        n_value = row["N"]
        if gpu != current_gpu and current_gpu is not None:
            lines.append("\\midrule")
        current_gpu = gpu

        vals = " & ".join(fmt(row[b]) for b in BASELINES)
        lines.append(f"{gpu} & {n_value} & {vals} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX: {path}")


def print_markdown_table(table):
    print("\n=== Markdown preview ===")
    print("| GPU | N | " + " | ".join(BASELINES) + " |")
    print("|---|---|" + "---|" * len(BASELINES))
    for row in table:
        print("| " + " | ".join([row["GPU"], row["N"], *[fmt(row[b]) for b in BASELINES]]) + " |")


def quality_check(table):
    all_vals = []
    below_1 = []

    for row in table:
        for baseline in BASELINES:
            v = row[baseline]
            if v is None:
                continue
            all_vals.append(v)
            if v < 1.0:
                below_1.append((row["GPU"], row["N"], baseline, v))

    print("\n=== Table quality check ===")
    if not all_vals:
        print("No valid values found.")
        return

    min_v = min(all_vals)
    max_v = max(all_vals)
    mean_v = fmean(all_vals)
    gt1 = sum(1 for v in all_vals if v > 1.0)
    print(f"cells={len(all_vals)}, min={min_v:.2f}, max={max_v:.2f}, mean={mean_v:.2f}")
    print(f"cells > 1.0: {gt1}/{len(all_vals)}")

    if below_1:
        print("Entries below 1.0 (sparsene slower than baseline) :")
        for gpu, n_value, baseline, v in below_1[:20]:
            print(f"  {gpu} N={n_value} vs {baseline}: {v:.2f}")
        if len(below_1) > 20:
            print(f"  ... and {len(below_1) - 20} more")
    else:
        print("All valid entries are >= 1.0.")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize sparsene speedup over baselines.")
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "jsons")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "csv" / "sparsene_vs_baselines_fp32_table.csv",
    )
    parser.add_argument(
        "--latex-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "latex" / "sparsene_vs_baselines_fp32_table.tex",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    table = collect_table(args.result_dir)
    print_markdown_table(table)
    quality_check(table)
    # write_csv(table, args.csv_out)
    # write_latex(table, args.latex_out)


if __name__ == "__main__":
    main()
