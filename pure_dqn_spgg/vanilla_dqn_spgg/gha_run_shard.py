from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

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


def _parse_jobs(raw: str) -> list[dict[str, object]]:
    jobs = json.loads(raw)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs_json must decode to a non-empty list")
    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one GitHub Actions shard of SPGG experiments")
    p.add_argument("--jobs-json", required=True, help="Compact JSON list of explicit {method, r, seed} jobs")
    p.add_argument("--output-root", required=True, help="Directory to write shard outputs into")
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--grid-size", type=int, required=True)
    p.add_argument("--save-frames-interval", type=int, default=0)
    p.add_argument("--tail-length", type=int, default=5000)
    p.add_argument("--history-len", type=int, default=3)
    p.add_argument("--update-prob", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--epsilon-min", type=float, default=0.0)
    p.add_argument("--dqn-init-mode", type=str, default="zero_last")
    p.add_argument("--greedy-tie-break", type=str, default="random")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--enable-adaptive-stop", action="store_true")
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


def _configure_threading(threads_per_worker: int) -> None:
    n = max(1, int(threads_per_worker))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = str(n)

    try:
        import torch

        torch.set_num_threads(n)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    _configure_threading(args.threads_per_worker)

    from batch_runner import _run_single

    shard_root = Path(args.output_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    jobs = _parse_jobs(args.jobs_json)
    run_output_root = shard_root / "runs"

    common = {
        "output_root": str(run_output_root),
        "iterations": args.iterations,
        "grid_size": args.grid_size,
        "save_frames_interval": args.save_frames_interval,
        "tail_length": args.tail_length,
        "history_len": args.history_len,
        "update_prob": args.update_prob,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "dqn_init_mode": args.dqn_init_mode,
        "greedy_tie_break": args.greedy_tie_break,
        "adaptive_stop": args.enable_adaptive_stop,
        "save_model": args.save_model,
        "threads_per_worker": args.threads_per_worker,
    }
    runner_jobs = [{**common, **job} for job in jobs]

    if args.workers <= 1:
        results = [_run_single(job) for job in runner_jobs]
    else:
        n_workers = min(int(args.workers), cpu_count(), len(runner_jobs))
        with Pool(n_workers) as pool:
            results = pool.map(_run_single, runner_jobs)

    results.sort(key=lambda row: (row["method"], float(row["r"]), int(row["seed"])))

    summary_path = shard_root / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "jobs": jobs,
        "common": common,
        "n_jobs": len(jobs),
        "n_ok": sum(1 for row in results if row["status"] == "ok"),
        "n_error": sum(1 for row in results if row["status"] != "ok"),
        "summary_csv": str(summary_path),
        "runs_root": str(run_output_root),
    }
    with (shard_root / "shard_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Shard complete: {summary_path}")


if __name__ == "__main__":
    main()
