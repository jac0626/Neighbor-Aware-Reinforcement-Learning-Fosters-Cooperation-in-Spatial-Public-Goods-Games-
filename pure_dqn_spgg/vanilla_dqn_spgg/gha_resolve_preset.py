from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = [
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
    "workers_per_runner",
    "threads_per_worker",
]


def _load_presets() -> dict[str, dict[str, str]]:
    preset_path = Path(__file__).with_name("gha_experiment_presets.json")
    with preset_path.open("r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve a GitHub Actions experiment preset")
    p.add_argument("--preset", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--r-values", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--grid-size", required=True)
    p.add_argument("--iterations", required=True)
    p.add_argument("--tail-length", required=True)
    p.add_argument("--history-len", required=True)
    p.add_argument("--save-frames-interval", required=True)
    p.add_argument("--update-prob", required=True)
    p.add_argument("--epsilon", required=True)
    p.add_argument("--epsilon-decay", required=True)
    p.add_argument("--epsilon-min", required=True)
    p.add_argument("--workers-per-runner", required=True)
    p.add_argument("--threads-per-worker", required=True)
    p.add_argument("--github-output", type=Path, help="Optional path to $GITHUB_OUTPUT")
    return p.parse_args()


def _custom_values(args: argparse.Namespace) -> dict[str, str]:
    return {
        "methods": args.methods,
        "r_values": args.r_values,
        "seeds": args.seeds,
        "shards": args.shards,
        "grid_size": args.grid_size,
        "iterations": args.iterations,
        "tail_length": args.tail_length,
        "history_len": args.history_len,
        "save_frames_interval": args.save_frames_interval,
        "update_prob": args.update_prob,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "workers_per_runner": args.workers_per_runner,
        "threads_per_worker": args.threads_per_worker,
    }


def resolve_config(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    presets = _load_presets()
    if args.preset == "custom":
        return "custom", _custom_values(args)

    if args.preset not in presets:
        raise ValueError(f"Unknown preset: {args.preset}")

    config = presets[args.preset]
    missing = [field for field in FIELDS if field not in config]
    if missing:
        raise ValueError(f"Preset {args.preset} is missing fields: {', '.join(missing)}")
    return args.preset, {field: str(config[field]) for field in FIELDS}


def main() -> None:
    args = parse_args()
    preset_name, config = resolve_config(args)

    payload = {"preset": preset_name, **config}

    if args.github_output is not None:
        with args.github_output.open("a") as f:
            for key, value in payload.items():
                f.write(f"{key}={value}\n")
        return

    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
