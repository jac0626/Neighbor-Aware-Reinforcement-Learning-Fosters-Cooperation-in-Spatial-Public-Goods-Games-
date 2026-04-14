"""Plot f_c vs r phase transition diagram from batch experiment results.

Reads summary.csv produced by batch_runner.py and generates a publication-
quality phase diagram with one curve per method.

Usage:
    python plot_phase_diagram.py --summary results/experiments/summary.csv \
        --output results/experiments/phase_diagram.pdf
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Visual style ────────────────────────────────────────────────────────
METHOD_STYLE = {
    "fermi":      {"color": "#2d2d2d", "marker": "o",  "label": "Fermi rule"},
    "tabular_q":  {"color": "#1f77b4", "marker": "s",  "label": "Tabular Q"},
    "dqn_self":   {"color": "#2ca02c", "marker": "^",  "label": "DQN (self)"},
    "dqn_local":  {"color": "#d62728", "marker": "D",  "label": "DQN (local)"},
    "dqn_vonn":   {"color": "#ff7f0e", "marker": "P",  "label": "DQN (vonn)"},
    "dqn_vonn_full": {"color": "#8c564b", "marker": "X", "label": "DQN (vonn-full)"},
    "dqn_wide":   {"color": "#9467bd", "marker": "h",  "label": "DQN (wide)"},
    "dqn_history": {"color": "#e377c2", "marker": "v", "label": "DQN (history)"},
}


def _read_summary(csv_path: Path) -> list[dict]:
    """Read summary.csv into a list of dicts."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def plot_phase_diagram(csv_path: Path, output_path: Path) -> None:
    """Create f_c vs r plot with mean ± std across seeds."""
    rows = _read_summary(csv_path)

    # Group by (method, r) → list of eq_coop_ratio
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        method = row["method"]
        r = float(row["r"])
        eq = float(row["eq_coop_ratio"])
        grouped[method][r].append(eq)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)

    for method in METHOD_STYLE:
        if method not in grouped:
            continue
        style = METHOD_STYLE[method]
        r_vals = sorted(grouped[method].keys())
        means = np.array([np.mean(grouped[method][r]) for r in r_vals])
        stds = np.array([np.std(grouped[method][r]) for r in r_vals])
        r_arr = np.array(r_vals)

        ax.plot(r_arr, means, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=1.8, markersize=7, zorder=3)
        ax.fill_between(r_arr, means - stds, means + stds,
                         alpha=0.15, color=style["color"], zorder=2)

    ax.set_xlabel(r"Return factor $r$", fontsize=13)
    ax.set_ylabel(r"Cooperation rate $f_c$", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Phase diagram saved to {output_path}")
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot f_c vs r phase diagram")
    p.add_argument("--summary", default="results/experiments/summary.csv",
                    help="Path to summary.csv from batch_runner")
    p.add_argument("--output", default="results/experiments/phase_diagram.pdf",
                    help="Output figure path (.pdf or .png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot_phase_diagram(Path(args.summary), Path(args.output))


if __name__ == "__main__":
    main()
