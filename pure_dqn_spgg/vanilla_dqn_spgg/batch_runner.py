"""Batch experiment runner for SPGG methods.

Orchestrates Fermi / Tabular-Q / DQN experiments across multiple
r values and random seeds.  Results are saved to a unified directory
structure and aggregated into a summary CSV.

Usage:
    python batch_runner.py --methods fermi tabular_q dqn_self dqn_local dqn_vonn dqn_wide dqn_history \
        --r-values 2.0 2.5 3.0 3.2 3.4 3.6 3.8 4.0 4.5 5.0 \
        --seeds 2026 2027 2028 2029 2030 \
        --output-root results/experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from fermi_spgg import FermiSPGGConfig, FermiSPGGEngine
from tabular_q_spgg import TabularQSPGGConfig, TabularQSPGGEngine
from run import VanillaDQNSPGGConfig, VanillaDQNSPGGEngine


DQN_METHOD_TO_STATE_MODE = {
    "dqn_self": "self",
    "dqn_local": "local",
    "dqn_vonn": "vonn",
    "dqn_vonn_full": "vonn_full",
    "dqn_wide": "wide",
    "dqn_history": "history",
}

SUPPORTED_METHODS = ("fermi", "tabular_q", *DQN_METHOD_TO_STATE_MODE.keys())


# ── Single experiment runner ────────────────────────────────────────────
def _run_single(args: dict) -> dict:
    """Run one (method, r, seed) combination. Returns a result dict."""
    method = args["method"]
    r = args["r"]
    seed = args["seed"]
    output_root = Path(args["output_root"])
    iterations = args["iterations"]
    grid_size = args["grid_size"]
    save_frames_interval = args["save_frames_interval"]
    update_prob = args.get("update_prob", 1.0)
    tail_length = int(args.get("tail_length", 5000))

    output_dir = output_root / method / f"r{r}_seed{seed}"

    try:
        if method == "fermi":
            cfg = FermiSPGGConfig(
                r=r, L=grid_size, iterations=iterations, seed=seed,
                save_frames_interval=save_frames_interval,
            )
            summary = FermiSPGGEngine(cfg).run(output_dir)

        elif method == "tabular_q":
            cfg = TabularQSPGGConfig(
                r=r, L=grid_size, iterations=iterations, seed=seed,
                save_frames_interval=save_frames_interval,
            )
            summary = TabularQSPGGEngine(cfg).run(output_dir)

        elif method in DQN_METHOD_TO_STATE_MODE:
            state_mode = DQN_METHOD_TO_STATE_MODE[method]
            cfg = VanillaDQNSPGGConfig(
                r=r, L=grid_size, iterations=iterations, seed=seed,
                state_mode=state_mode,
                history_len=args.get("history_len", 3),
                adaptive_stop=args.get("adaptive_stop", False),
                save_frames_interval=save_frames_interval,
                epsilon=args.get("epsilon", 0.5),
                epsilon_decay=args.get("epsilon_decay", 0.9995),
                epsilon_min=args.get("epsilon_min", 0.0),
                dqn_init_mode=args.get("dqn_init_mode", "zero_last"),
                greedy_tie_break=args.get("greedy_tie_break", "random"),
                update_prob=update_prob,
                save_model=args.get("save_model", True),
                deterministic_cpu=True,
            )
            summary = VanillaDQNSPGGEngine(cfg).run(output_dir)

        else:
            raise ValueError(f"Unknown method: {method}")

        import h5py
        h5_path = output_dir / "data" / "experiment_data.h5"
        with h5py.File(h5_path, "r") as f:
            coop_hist = f["coop_rate_history"][:]
            payoff_hist = f["avg_payoff_history"][:] if "avg_payoff_history" in f else None
            payoff_norm_hist = f["avg_payoff_norm_history"][:] if "avg_payoff_norm_history" in f else None

        tail = min(max(1, tail_length), len(coop_hist))
        eq_coop = float(np.mean(coop_hist[-tail:]))
        eq_std = float(np.std(coop_hist[-tail:]))
        eq_payoff = float(np.mean(payoff_hist[-tail:])) if payoff_hist is not None else float("nan")
        eq_payoff_norm = float(np.mean(payoff_norm_hist[-tail:])) if payoff_norm_hist is not None else float("nan")

        result = {
            "method": method,
            "state_mode": DQN_METHOD_TO_STATE_MODE.get(method, method),
            "dqn_init_mode": cfg.dqn_init_mode if method in DQN_METHOD_TO_STATE_MODE else "",
            "greedy_tie_break": cfg.greedy_tie_break if method in DQN_METHOD_TO_STATE_MODE else "",
            "r": r,
            "seed": seed,
            "final_coop_ratio": summary["final_coop_ratio"],
            "final_avg_payoff": summary.get("final_avg_payoff", float("nan")),
            "final_avg_payoff_norm": summary.get("final_avg_payoff_norm", float("nan")),
            "eq_coop_ratio": eq_coop,
            "eq_coop_std": eq_std,
            "eq_avg_payoff": eq_payoff,
            "eq_avg_payoff_norm": eq_payoff_norm,
            "total_steps": summary["total_steps"],
            "status": "ok",
            "error": "",
        }
        print(f"✓ {method} r={r} seed={seed} → eq_coop={eq_coop:.4f} eq_payoff={eq_payoff:.4f}")
        return result

    except Exception as e:
        print(f"✗ {method} r={r} seed={seed} → ERROR: {e}")
        traceback.print_exc()
        return {
            "method": method,
            "state_mode": DQN_METHOD_TO_STATE_MODE.get(method, method),
            "dqn_init_mode": args.get("dqn_init_mode", "") if method in DQN_METHOD_TO_STATE_MODE else "",
            "greedy_tie_break": args.get("greedy_tie_break", "") if method in DQN_METHOD_TO_STATE_MODE else "",
            "r": r,
            "seed": seed,
            "final_coop_ratio": float("nan"),
            "final_avg_payoff": float("nan"),
            "final_avg_payoff_norm": float("nan"),
            "eq_coop_ratio": float("nan"),
            "eq_coop_std": float("nan"),
            "eq_avg_payoff": float("nan"),
            "eq_avg_payoff_norm": float("nan"),
            "total_steps": 0,
            "status": "error",
            "error": str(e),
        }


# ── CLI & orchestration ────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch SPGG experiment runner")
    p.add_argument("--output-root", default="results/experiments")
    p.add_argument(
        "--methods", nargs="+",
        default=["fermi", "tabular_q", "dqn_self", "dqn_local", "dqn_wide"],
        help=f"Methods to run. Supported: {', '.join(SUPPORTED_METHODS)}",
    )
    p.add_argument(
        "--r-values", nargs="+", type=float,
        default=[2.0, 2.5, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0],
        help="Return factor values to sweep",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int,
        default=[2026, 2027, 2028, 2029, 2030],
        help="Random seeds for independent runs",
    )
    p.add_argument("--grid-size", type=int, default=100)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--save-frames-interval", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tail-length", type=int, default=5000)
    p.add_argument("--history-len", type=int, default=3)
    p.add_argument("--update-prob", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--epsilon-min", type=float, default=0.0)
    p.add_argument("--dqn-init-mode", type=str, default="zero_last")
    p.add_argument("--greedy-tie-break", type=str, default="random")
    p.add_argument("--enable-adaptive-stop", action="store_true")
    p.add_argument("--no-save-model", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []

    for method in args.methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method {method}. Supported methods: {', '.join(SUPPORTED_METHODS)}")
        for r in args.r_values:
            for seed in args.seeds:
                jobs.append({
                    "method": method,
                    "r": r,
                    "seed": seed,
                    "output_root": str(output_root),
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
                    "save_model": not args.no_save_model,
                })

    total = len(jobs)
    print(f"Running {total} experiments: {len(args.methods)} methods × "
          f"{len(args.r_values)} r-values × {len(args.seeds)} seeds")
    print(f"Workers: {args.workers}")
    print()

    # Execute
    if args.workers <= 1:
        results = [_run_single(job) for job in jobs]
    else:
        n_workers = min(args.workers, cpu_count(), total)
        with Pool(n_workers) as pool:
            results = pool.map(_run_single, jobs)

    # Write summary CSV
    csv_path = output_root / "summary.csv"
    fieldnames = [
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
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    manifest_path = output_root / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "methods": args.methods,
                "r_values": args.r_values,
                "seeds": args.seeds,
                "grid_size": args.grid_size,
                "iterations": args.iterations,
                "tail_length": args.tail_length,
                "history_len": args.history_len,
                "update_prob": args.update_prob,
                "epsilon": args.epsilon,
                "epsilon_decay": args.epsilon_decay,
                "epsilon_min": args.epsilon_min,
                "dqn_init_mode": args.dqn_init_mode,
                "greedy_tie_break": args.greedy_tie_break,
                "adaptive_stop": args.enable_adaptive_stop,
                "save_model": not args.no_save_model,
            },
            f,
            indent=2,
        )

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = total - n_ok
    print(f"\n{'='*60}")
    print(f"Summary saved to {csv_path}")
    print(f"Manifest saved to {manifest_path}")
    print(f"Completed: {n_ok}/{total}  Errors: {n_err}")


if __name__ == "__main__":
    main()
