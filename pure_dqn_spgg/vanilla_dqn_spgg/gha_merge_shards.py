from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plot_phase_diagram import plot_phase_diagram


SUMMARY_FIELDS = [
    "method",
    "state_mode",
    "r",
    "seed",
    "final_coop_ratio",
    "final_avg_payoff",
    "final_avg_payoff_norm",
    "eq_coop_ratio",
    "eq_coop_std",
    "eq_avg_payoff",
    "eq_avg_payoff_norm",
    "total_steps",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge GitHub Actions shard summaries")
    p.add_argument("--input-dir", required=True, help="Directory containing downloaded shard artifacts")
    p.add_argument("--output-dir", required=True, help="Directory to write merged outputs into")
    p.add_argument("--plot-output", help="Optional phase diagram path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(input_dir.rglob("summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No shard summary.csv files found under {input_dir}")

    rows: list[dict[str, str]] = []
    manifests = []

    for summary_file in summary_files:
        with summary_file.open("r", newline="") as f:
            rows.extend(csv.DictReader(f))

        manifest_file = summary_file.with_name("shard_manifest.json")
        if manifest_file.exists():
            with manifest_file.open("r") as f:
                manifests.append(json.load(f))

    rows.sort(key=lambda row: (row["method"], float(row["r"]), int(row["seed"])))

    merged_summary = output_dir / "summary.csv"
    with merged_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    merged_manifest = {
        "n_shards": len(summary_files),
        "n_rows": len(rows),
        "summary_files": [str(path) for path in summary_files],
        "shard_manifests": manifests,
    }
    with (output_dir / "merged_manifest.json").open("w") as f:
        json.dump(merged_manifest, f, indent=2)

    if args.plot_output:
        plot_phase_diagram(merged_summary, Path(args.plot_output))

    print(f"Merged summary saved to {merged_summary}")


if __name__ == "__main__":
    main()
