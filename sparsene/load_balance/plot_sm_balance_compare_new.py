#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


TIME_RE = re.compile(r"mykernel_time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
SM_RE = re.compile(r"SM\[(\d+)\]\s+accumulated_cycles=([0-9]+)")


def parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    time_match = TIME_RE.search(text)
    if time_match is None:
        raise ValueError(f"Cannot find mykernel_time in {log_path}")
    kernel_time_ms = float(time_match.group(1))

    sm_pairs = SM_RE.findall(text)
    if not sm_pairs:
        raise ValueError(f"Cannot find SM accumulated_cycles in {log_path}")

    max_idx = max(int(idx) for idx, _ in sm_pairs)
    cycles = np.zeros(max_idx + 1, dtype=np.float64)
    for idx_str, cyc_str in sm_pairs:
        cycles[int(idx_str)] = float(cyc_str)

    max_cycle = float(np.max(cycles))
    if max_cycle <= 0.0:
        raise ValueError(f"Invalid max cycle in {log_path}")

    # Normalize to per-SM utilization: 1.0 means close to fully busy.
    execution = cycles / max_cycle
    idle = 1.0 - execution
    return {
        "time_ms": kernel_time_ms,
        "execution": execution,
        "idle": idle,
    }


def parse_csv_arg(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_matrix_name(name: str):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def infer_strategy_key(with_label: str, with_log_path: Path):
    token = f"{with_label} {with_log_path.name}".lower()
    if "strict" in token:
        return "speedup_strict_lb"
    if "multi" in token or "bind" in token:
        return "speedup_multi_binding"
    return "speedup_multi_binding"


def load_speedup_from_ablation(ablation_data: dict, matrix_name: str, feat_n: int, strategy_key: str):
    n_key = str(feat_n)

    if matrix_name in ablation_data and n_key in ablation_data[matrix_name]:
        speedup = ablation_data[matrix_name][n_key].get(strategy_key)
        if speedup is not None:
            return float(speedup)

    norm_target = normalize_matrix_name(matrix_name)
    for key, value in ablation_data.items():
        if normalize_matrix_name(key) == norm_target and n_key in value:
            speedup = value[n_key].get(strategy_key)
            if speedup is not None:
                return float(speedup)

    raise ValueError(
        f"Cannot find {strategy_key} for matrix={matrix_name}, N={feat_n} in ablation json"
    )


def choose_ylim(
    execution: np.ndarray,
    prefer_tight=False,
    default_ymin=0.4,
    tight_ymin=0.99,
    tight_threshold=0.97,
):
    if prefer_tight and float(np.min(execution)) > tight_threshold:
        return tight_ymin, 1.0
    lower = max(default_ymin, float(np.min(execution)) - 0.05)
    return lower, 1.0


def draw_row(
    fig,
    outer_grid,
    row_idx,
    matrix_name,
    no_bal_data,
    bal_data,
    speedup,
    without_label="w/o balance",
    with_label="w/ balance",
    default_ymin=0.4,
    tight_ymin=0.99,
    tight_threshold=0.97,
    with_ymin=0.7,
    panel_label=None,
    panel_label_offset=0.08,
    right_ylabel_pad=4,
):
    light_green = "#e07a2f"
    dark_green = "#6fa8dc"
    blue_idle = "#b8c4cc"

    ax_left = fig.add_subplot(outer_grid[row_idx, 0])
    ax_left.set_facecolor("white")

    bars = ax_left.bar(
        [0, 1],
        [1.0, speedup],
        color=[light_green, dark_green],
        width=0.62,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_left.set_xticks([0, 1], [without_label, with_label], fontsize=11.5)
    ax_left.tick_params(axis="y", labelsize=14)
    # ax_left.set_xlabel(matrix_name, fontsize=18, fontweight="bold")
    ax_left.set_ylabel("Normalized Speedup", fontsize=15)
    y_top = max(1.8, speedup * 1.15)
    ax_left.set_ylim(0.0, y_top)
    ax_left.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    gain_pct = (speedup - 1.0) * 100.0
    ax_left.text(
        bars[1].get_x() + bars[1].get_width() / 2.0,
        bars[1].get_height() + y_top * 0.03,
        f"+{gain_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=13,
    )
    #              ，wspace  ，    
    right = outer_grid[row_idx, 1].subgridspec(1, 2, wspace=0.08)
    ax_no = fig.add_subplot(right[0, 0])
    ax_yes = fig.add_subplot(right[0, 1], sharey=ax_no)

    for ax, title, item, prefer_tight, is_with_balance in [
        (ax_no, without_label, no_bal_data, False, False),
        (ax_yes, with_label, bal_data, True, True),
    ]:
        ax.set_facecolor("white")
        x = np.arange(len(item["execution"]))
        exec_vals = item["execution"]
        idle_vals = item["idle"]

        ax.bar(
            x,
            exec_vals,
            width=1.0,
            align="edge",
            color=light_green,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            label="Execution",
        )
        ax.bar(
            x,
            idle_vals,
            width=1.0,
            align="edge",
            bottom=exec_vals,
            color=blue_idle,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            label="Idle",
        )

        # Keep the two right subplots on the same visible y-range.
        y0, y1 = with_ymin, 1.0
        ax.set_ylim(y0, y1)
        ax.set_xlim(0, len(x))
        ax.margins(x=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("")
        if is_with_balance:
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Normalized \nSM Utilization", fontsize=15, labelpad=right_ylabel_pad)
        #       "w/o balance" "w/ balance"
        ax.text(0.5, -0.13, title, transform=ax.transAxes, ha="center", va="top", fontsize=15, fontweight="bold")
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0.0,
            ncol=2,
            frameon=False,
            fontsize=15,
        )
        ax.tick_params(axis="both", colors="black", labelsize=13)

    ax_left.tick_params(axis="both", colors="black")

    if panel_label:
        pos_left = ax_left.get_position()
        pos_no = ax_no.get_position()
        pos_yes = ax_yes.get_position()
        row_center_x = (pos_left.x0 + pos_yes.x1) / 2.0
        row_bottom_y = min(pos_left.y0, pos_no.y0, pos_yes.y0)
        fig.text(
            row_center_x,
            row_bottom_y - panel_label_offset,
            panel_label,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

    return ax_left


def add_row_separators(fig, left_axes):
    if len(left_axes) < 2:
        return
    for i in range(len(left_axes) - 1):
        cur_pos = left_axes[i].get_position()
        nxt_pos = left_axes[i + 1].get_position()
        y = (cur_pos.y0 + nxt_pos.y1) / 2.0
        fig.add_artist(
            plt.Line2D(
                [0.07, 0.9],
                [y, y],
                transform=fig.transFigure,
                linestyle="-.",
                color="black",
                linewidth=1.2,
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot SM load comparison for w/o balance vs w/ multi-bind."
    )
    parser.add_argument(
        "--matrices",
        type=str,
        default="mip1",
        help="Comma-separated matrix names. Supports one or two rows.",
    )
    parser.add_argument(
        "--without-logs",
        type=str,
        default="./mip1_no_balance.log",
        help="Comma-separated log paths for w/o balance.",
    )
    parser.add_argument(
        "--with-logs",
        type=str,
        default="./mip1_multi_bind.log",
        help="Comma-separated log paths for w/ balance (multi-bind).",
    )
    parser.add_argument(
        "--ablation-json",
        type=str,
        default="./ablation_merged.json",
        help="Ablation summary JSON used to fetch left-panel speedup.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=512,
        help="Feature dimension N used to query ablation speedup.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./mip1_sm_balance_compare.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--default-ymin",
        type=float,
        default=0.4,
        help="Lower bound for right-panel y-axis in non-tight mode.",
    )
    parser.add_argument(
        "--tight-ymin",
        type=float,
        default=0.99,
        help="Lower bound for right-panel y-axis in tight mode.",
    )
    parser.add_argument(
        "--tight-threshold",
        type=float,
        default=0.97,
        help="Enable tight mode when min execution is above this threshold.",
    )
    parser.add_argument(
        "--with-ymin",
        type=float,
        default=0.7,
        help="Lower bound for both right-panel charts in each row.",
    )
    parser.add_argument(
        "--panel-label-offset",
        type=float,
        default=0.08,
        help="Vertical offset for (a)/(b) labels below each row in figure coords.",
    )
    parser.add_argument(
        "--row-hspace",
        type=float,
        default=0.62,
        help="Vertical spacing between row 1 and row 2 (outer gridspec hspace).",
    )
    parser.add_argument(
        "--group-wspace",
        type=float,
        default=0.22,
        help="Horizontal spacing between left speedup plot and right two SM plots (outer gridspec wspace).",
    )
    parser.add_argument(
        "--right-ylabel-pad",
        type=float,
        default=4,
        help="Padding for the right-side left subplot y-axis label (Normalized SM Utilization).",
    )
    parser.add_argument(
        "--without-labels",
        type=str,
        default="w/o balance",
        help="Comma-separated row labels for baseline strategy.",
    )
    parser.add_argument(
        "--with-labels",
        type=str,
        default="w/ balance",
        help="Comma-separated row labels for balanced strategy.",
    )
    parser.add_argument(
        "--panel-labels",
        type=str,
        default="",
        help="Optional comma-separated row labels shown as (a)/(b) captions. Defaults to matrix names.",
    )
    args = parser.parse_args()

    matrices = parse_csv_arg(args.matrices)
    no_logs = [Path(p) for p in parse_csv_arg(args.without_logs)]
    yes_logs = [Path(p) for p in parse_csv_arg(args.with_logs)]
    without_labels = parse_csv_arg(args.without_labels)
    with_labels = parse_csv_arg(args.with_labels)

    with open(args.ablation_json, "r", encoding="utf-8") as f:
        ablation_data = json.load(f)

    def expand_items(items, n, name):
        if len(items) == 1 and n > 1:
            return items * n
        if len(items) != n:
            raise ValueError(f"Length of {name} must be 1 or match number of rows ({n})")
        return items

    if not (len(matrices) == len(no_logs) == len(yes_logs)):
        raise ValueError("Lengths of --matrices, --without-logs, --with-logs must match")
    if len(matrices) == 0 or len(matrices) > 2:
        raise ValueError("Please provide one or two matrices")

    without_labels = expand_items(without_labels, len(matrices), "--without-labels")
    with_labels = expand_items(with_labels, len(matrices), "--with-labels")
    with_strategy_keys = [infer_strategy_key(with_labels[i], yes_logs[i]) for i in range(len(matrices))]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"

    nrows = len(matrices)
    if args.panel_labels.strip():
        panel_label_texts = expand_items(parse_csv_arg(args.panel_labels), nrows, "--panel-labels")
    else:
        panel_label_texts = matrices
    panel_labels = [f"({chr(ord('a') + i)}){panel_label_texts[i]}" for i in range(nrows)]
    fig = plt.figure(figsize=(13.2, 3.2 * nrows), dpi=220, facecolor="white")
    # wspace   ，        ；hspace   ，        
    outer = fig.add_gridspec(
        nrows,
        2,
        width_ratios=[0.8, 3.2],
        hspace=args.row_hspace,
        wspace=args.group_wspace,
    )

    left_axes = []
    for i, matrix_name in enumerate(matrices):
        no_data = parse_log(no_logs[i])
        yes_data = parse_log(yes_logs[i])
        left_axes.append(
            draw_row(
                fig,
                outer,
                i,
                matrix_name,
                no_data,
                yes_data,
                speedup=load_speedup_from_ablation(
                    ablation_data,
                    matrix_name,
                    args.N,
                    with_strategy_keys[i],
                ),
                without_label=without_labels[i],
                with_label=with_labels[i],
                default_ymin=args.default_ymin,
                tight_ymin=args.tight_ymin,
                tight_threshold=args.tight_threshold,
                with_ymin=args.with_ymin,
                panel_label=panel_labels[i],
                panel_label_offset=args.panel_label_offset,
                right_ylabel_pad=args.right_ylabel_pad,
            )
        )

    # add_row_separators(fig, left_axes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
