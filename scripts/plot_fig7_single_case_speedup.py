import argparse
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

title_font_size = 20
label_font_size = 20
legend_font_size = 18
tick_font_size = 20
ytick_font_size = 20


DEVICE_TO_FILE_FP32 = {
    "A100": "a100_fp32_merged.json",
    "H100": "h800_fp32_merged.json",
    "4090": "4090_fp32_merged.json",
}

MATRIX_DICT = {
    "YeastH": "YH",
    "OVCAR-8H": "OH",
    "Yeast": "Yt",
    "DD": "DD",
    "web-BerkStan": "WB",
    "reddit": "reddit",
    "ddi": "ddi",
    "protein": "protein",
    "pwtk": "M1",
    "rma10": "M2",
    "shipsec1": "M3",
    "scircuit": "M4",
    "conf5_4-8x8-10": "M5",
    "mc2depi": "M6",
    "webbase-1M": "M7",
    "mac_econ_fwd500": "M8",
    "pdb1HYS": "M9",
    "consph": "M10",
    "cant": "M11",
    "cop20k_A": "M12",
    "eu-2005": "M13",
    "Si41Ge41H72": "M14",
    "Ga41As41H72": "M15",
    "mip1": "M16",
    "dc2": "M17",
    "ASIC_680k": "M18",
}

METHOD_ORDER = [
    "cusparse",
    "sputnik",
    "DTC-SPMM",
    "ACC-SPMM",
    "FlashSparse",
    "SparseTIR",
    "sparsene",
]

METHOD_COLORS = {
    "cusparse": "#2ca02c",
    "sputnik": "#9467bd",
    "DTC-SPMM": "#8c564b",
    "ACC-SPMM": "#e377c2",
    "FlashSparse": "#1f77b4",
    "SparseTIR": "#d62728",
    "sparsene": "#ff7f0e",
}


def load_device_data(result_dir: Path, device: str):
    file_name = DEVICE_TO_FILE_FP32[device]
    if not (result_dir / file_name).exists():
        return {}
    with open(result_dir / file_name, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_geomean(values):
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def compute_metrics_for_choice(data, n_value: str):
    sparsene_speedups = []
    sparsene_advantages = []
    covered = 0

    for matrix_name, by_n in data.items():
        if n_value not in by_n:
            continue
        vals = by_n[n_value]
        if "cusparse" not in vals or "sparsene" not in vals:
            continue

        cusp = vals["cusparse"]
        if cusp <= 0 or vals["sparsene"] <= 0:
            continue

        covered += 1
        s_speed = cusp / vals["sparsene"]
        sparsene_speedups.append(s_speed)

        competitors = []
        for method in METHOD_ORDER:
            if method in ("cusparse", "sparsene"):
                continue
            t = vals.get(method)
            if t is not None and t > 0:
                competitors.append(cusp / t)

        if competitors:
            sparsene_advantages.append(s_speed - max(competitors))

    if not sparsene_speedups:
        return None

    geo = safe_geomean(sparsene_speedups)
    mean_adv = float(fmean(sparsene_advantages)) if sparsene_advantages else -999.0
    std_s = float(pstdev(sparsene_speedups)) if len(sparsene_speedups) > 1 else 0.0

    # High score means strong and stable sparsene lead with decent coverage.
    score = (mean_adv * 0.7) + (math.log(geo) * 0.3) - (0.05 * std_s) + (0.01 * covered)

    return {
        "coverage": covered,
        "sparsene_geomean_vs_cusparse": geo,
        "mean_advantage_vs_best_competitor": mean_adv,
        "sparsene_std": std_s,
        "score": score,
    }


def rank_all_choices(result_dir: Path):
    rows = []
    for device in DEVICE_TO_FILE_FP32:
        data = load_device_data(result_dir, device)
        for n_value in ["128", "256", "512"]:
            m = compute_metrics_for_choice(data, n_value)
            if m is None:
                continue
            m["device"] = device
            m["N"] = n_value
            rows.append(m)

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def plot_single_case(data, device: str, n_value: str, output: Path):
    import matplotlib.pyplot as plt

    matrices = [m for m in MATRIX_DICT if m in data and n_value in data[m]]
    if not matrices:
        raise ValueError(f"No matrices found for device={device}, N={n_value}")

    fig, ax = plt.subplots(figsize=(22, 4.5))
    bar_width = 0.8 / len(METHOD_ORDER)
    labeled_methods = set()

    xticks = []
    xlabels = []

    for m_idx, matrix_name in enumerate(matrices):
        vals = data[matrix_name][n_value]
        cusp = vals.get("cusparse")
        if cusp is None or cusp <= 0:
            continue

        for j, method in enumerate(METHOD_ORDER):
            x = m_idx + j * bar_width
            if method == "cusparse":
                speedup = 1.0
            else:
                t = vals.get(method)
                if t is None or t <= 0:
                    continue
                speedup = cusp / t

            alpha = 1.0 if method == "sparsene" else 0.88
            label = method if method not in labeled_methods else ""
            ax.bar(
                x,
                speedup,
                width=bar_width,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.4,
                alpha=alpha,
                label=label,
            )
            labeled_methods.add(method)

        xticks.append(m_idx + (len(METHOD_ORDER) * bar_width) / 2.0)
        xlabels.append(MATRIX_DICT.get(matrix_name, matrix_name))

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    # ax.set_title(f"FP32 speedup over cuSPARSE on {device} (N={n_value})", fontsize=title_font_size)
    ax.set_ylabel("Normalized speedup", fontsize=label_font_size)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=tick_font_size)
    ax.tick_params(axis="y", labelsize=ytick_font_size)
    ax.set_xlim(-0.35, len(matrices) - 0.05)

    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l and l not in by_label:
            by_label[l] = h
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=7,
        fontsize=legend_font_size,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(f"Saved figure: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot one device+N speedup chart and rank all choices.")
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "jsons")
    parser.add_argument("--device", choices=["A100", "H100", "4090"], default="A100")
    parser.add_argument("--N", choices=["128", "256", "512"], default="512")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "fig" / "fig7.pdf",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only print ranking for all device/N choices, do not generate plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ranking = rank_all_choices(args.result_dir)
    print("=== Recommendation ranking (higher score is better) ===")
    for i, row in enumerate(ranking[:9], start=1):
        print(
            f"{i:>2}. {row['device']} N={row['N']} | score={row['score']:.4f} | "
            f"coverage={row['coverage']} | "
            f"geo(sparsene/cuSPARSE)={row['sparsene_geomean_vs_cusparse']:.3f} | "
            f"mean_adv_vs_best={row['mean_advantage_vs_best_competitor']:.3f}"
        )

    if args.scan_only:
        return

    data = load_device_data(args.result_dir, args.device)
    plot_single_case(data, args.device, args.N, args.output)


if __name__ == "__main__":
    main()
