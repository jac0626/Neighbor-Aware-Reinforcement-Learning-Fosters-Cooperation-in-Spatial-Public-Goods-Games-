from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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


def build_matrix(methods: list[str], r_values: list[float], seeds: list[int], shards: int) -> dict[str, list[dict[str, object]]]:
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

    include = []
    for shard_index, bucket in enumerate(buckets):
        if not bucket:
            continue
        include.append(
            {
                "shard_index": shard_index,
                "shard_name": f"shard-{shard_index:02d}",
                "job_count": len(bucket),
                "jobs_json": json.dumps(bucket, separators=(",", ":")),
            }
        )
    return {"include": include}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a GitHub Actions shard matrix for SPGG experiments")
    p.add_argument("--methods", required=True, help="Space/comma separated methods")
    p.add_argument("--r-values", required=True, help="Space/comma separated r values")
    p.add_argument("--seeds", required=True, help="Space/comma separated seeds")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--github-output", type=Path, help="Optional path to $GITHUB_OUTPUT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matrix = build_matrix(
        methods=_parse_methods(args.methods),
        r_values=_parse_r_values(args.r_values),
        seeds=_parse_seeds(args.seeds),
        shards=args.shards,
    )
    matrix_json = json.dumps(matrix, separators=(",", ":"))

    if args.github_output is not None:
        with args.github_output.open("a") as f:
            f.write(f"matrix={matrix_json}\n")
            f.write(f"shard_count={len(matrix['include'])}\n")
            f.write(f"job_count={sum(item['job_count'] for item in matrix['include'])}\n")
        return

    print(matrix_json)


if __name__ == "__main__":
    main()
