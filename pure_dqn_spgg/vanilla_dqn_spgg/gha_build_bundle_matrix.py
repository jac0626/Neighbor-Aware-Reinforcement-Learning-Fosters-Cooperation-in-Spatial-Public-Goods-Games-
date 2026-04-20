from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_FIELDS = [
    "bundle",
    "methods",
    "r_values",
    "seeds",
    "shards",
    "grid_size",
    "iterations",
    "tail_length",
    "history_len",
    "save_frames_interval",
    "update_prob",
    "epsilon",
    "epsilon_decay",
    "epsilon_min",
    "dqn_init_mode",
    "greedy_tie_break",
    "workers_per_runner",
    "threads_per_worker",
    "adaptive_stop",
    "save_models",
    "upload_raw_results",
]


def _parse_tokens(raw: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", raw.strip()) if token]


def _parse_methods(raw: str) -> list[str]:
    methods = _parse_tokens(raw)
    if not methods:
        raise ValueError("methods cannot be empty")
    return methods


def _parse_r_values(raw: str) -> list[float]:
    values = [float(token) for token in _parse_tokens(raw)]
    if not values:
        raise ValueError("r_values cannot be empty")
    return values


def _parse_seeds(raw: str) -> list[int]:
    values = [int(token) for token in _parse_tokens(raw)]
    if not values:
        raise ValueError("seeds cannot be empty")
    return values


def _load_plan(path: Path, profile: str) -> list[dict[str, str]]:
    with path.open("r") as f:
        raw = json.load(f)

    if profile not in raw:
        raise ValueError(f"Unknown plan profile: {profile}")

    bundles = raw[profile]
    if not isinstance(bundles, list) or not bundles:
        raise ValueError(f"Plan profile {profile} must be a non-empty list")

    for bundle in bundles:
        missing = [field for field in REQUIRED_FIELDS if field not in bundle]
        if missing:
            raise ValueError(
                f"Bundle {bundle.get('bundle', '<unnamed>')} is missing fields: {', '.join(missing)}"
            )
    return bundles


def _bucket_jobs(methods: list[str], r_values: list[float], seeds: list[int], shards: int) -> list[list[dict[str, object]]]:
    jobs = [
        {"method": method, "r": r_value, "seed": seed}
        for method in methods
        for r_value in r_values
        for seed in seeds
    ]
    if not jobs:
        raise ValueError("no jobs to schedule")

    shard_count = max(1, min(int(shards), len(jobs)))
    buckets = [[] for _ in range(shard_count)]
    for idx, job in enumerate(jobs):
        buckets[idx % shard_count].append(job)
    return [bucket for bucket in buckets if bucket]


def build_matrix(plan_path: Path, profile: str) -> dict[str, list[dict[str, object]]]:
    bundles = _load_plan(plan_path, profile)
    include: list[dict[str, object]] = []

    for bundle in bundles:
        bundle_name = str(bundle["bundle"])
        methods = _parse_methods(str(bundle["methods"]))
        r_values = _parse_r_values(str(bundle["r_values"]))
        seeds = _parse_seeds(str(bundle["seeds"]))
        buckets = _bucket_jobs(methods, r_values, seeds, int(bundle["shards"]))

        for shard_index, bucket in enumerate(buckets):
            include.append(
                {
                    "bundle_name": bundle_name,
                    "bundle_description": str(bundle.get("description", "")),
                    "shard_index": shard_index,
                    "shard_name": f"shard-{shard_index:02d}",
                    "job_count": len(bucket),
                    "jobs_json": json.dumps(bucket, separators=(",", ":")),
                    "grid_size": str(bundle["grid_size"]),
                    "iterations": str(bundle["iterations"]),
                    "tail_length": str(bundle["tail_length"]),
                    "history_len": str(bundle["history_len"]),
                    "save_frames_interval": str(bundle["save_frames_interval"]),
                    "update_prob": str(bundle["update_prob"]),
                    "epsilon": str(bundle["epsilon"]),
                    "epsilon_decay": str(bundle["epsilon_decay"]),
                    "epsilon_min": str(bundle["epsilon_min"]),
                    "dqn_init_mode": str(bundle["dqn_init_mode"]),
                    "greedy_tie_break": str(bundle["greedy_tie_break"]),
                    "workers_per_runner": str(bundle["workers_per_runner"]),
                    "threads_per_worker": str(bundle["threads_per_worker"]),
                    "adaptive_stop": str(bundle["adaptive_stop"]).lower(),
                    "save_models": str(bundle["save_models"]).lower(),
                    "upload_raw_results": str(bundle["upload_raw_results"]).lower(),
                }
            )

    return {"include": include}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a GitHub Actions matrix from a paper experiment plan")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--profile", required=True)
    p.add_argument("--github-output", type=Path, help="Optional path to $GITHUB_OUTPUT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matrix = build_matrix(args.plan, args.profile)
    matrix_json = json.dumps(matrix, separators=(",", ":"))
    bundle_count = len({item["bundle_name"] for item in matrix["include"]})
    shard_count = len(matrix["include"])
    job_count = sum(int(item["job_count"]) for item in matrix["include"])

    if args.github_output is not None:
        with args.github_output.open("a") as f:
            f.write(f"matrix={matrix_json}\n")
            f.write(f"bundle_count={bundle_count}\n")
            f.write(f"shard_count={shard_count}\n")
            f.write(f"job_count={job_count}\n")
        return

    print(matrix_json)


if __name__ == "__main__":
    main()
