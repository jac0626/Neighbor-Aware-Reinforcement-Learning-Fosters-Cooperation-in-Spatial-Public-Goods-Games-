from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from plot_phase_diagram import plot_phase_diagram


SUMMARY_FIELDS = [
    "method",
    "state_mode",
    "dqn_init_mode",
    "greedy_tie_break",
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
    p = argparse.ArgumentParser(description="Merge bundle-based GitHub Actions shard summaries")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(input_dir.rglob("summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No shard summary.csv files found under {input_dir}")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    grouped_manifests: dict[str, list[dict[str, object]]] = defaultdict(list)

    for summary_file in summary_files:
        manifest_file = summary_file.with_name("shard_manifest.json")
        bundle_name = "default"
        manifest_data: dict[str, object] | None = None
        if manifest_file.exists():
            with manifest_file.open("r") as f:
                manifest_data = json.load(f)
            bundle_name = str(manifest_data.get("bundle") or "default")
            grouped_manifests[bundle_name].append(manifest_data)

        with summary_file.open("r", newline="") as f:
            grouped_rows[bundle_name].extend(csv.DictReader(f))

    top_manifest: dict[str, object] = {
        "n_bundles": len(grouped_rows),
        "bundles": {},
    }

    for bundle_name, rows in sorted(grouped_rows.items()):
        rows.sort(key=lambda row: (row["method"], float(row["r"]), int(row["seed"])))
        bundle_dir = output_dir / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        merged_summary = bundle_dir / "summary.csv"
        with merged_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        bundle_manifest = {
            "bundle": bundle_name,
            "n_rows": len(rows),
            "shard_manifests": grouped_manifests.get(bundle_name, []),
        }
        with (bundle_dir / "merged_manifest.json").open("w") as f:
            json.dump(bundle_manifest, f, indent=2)

        r_values = sorted({float(row["r"]) for row in rows})
        methods = sorted({row["method"] for row in rows})
        if len(r_values) > 1 and len(methods) > 1:
            plot_phase_diagram(merged_summary, bundle_dir / "phase_diagram.png")

        top_manifest["bundles"][bundle_name] = {
            "summary_csv": str(merged_summary),
            "merged_manifest": str(bundle_dir / "merged_manifest.json"),
            "phase_diagram": str(bundle_dir / "phase_diagram.png") if (bundle_dir / "phase_diagram.png").exists() else "",
            "n_rows": len(rows),
        }

    with (output_dir / "paper_full_manifest.json").open("w") as f:
        json.dump(top_manifest, f, indent=2)

    print(f"Merged bundle outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
