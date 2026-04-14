"""Fermi rule baseline for Spatial Public Goods Game.

Classical evolutionary dynamics: agents copy a random neighbor's strategy
with probability given by the Fermi function of payoff difference.
No learning — purely imitation-driven.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from spgg_utils import normalize_payoff_fixed, overlap5


# ── Configuration ───────────────────────────────────────────────────────
@dataclass(slots=True)
class FermiSPGGConfig:
    # SPGG environment (same defaults as DQN)
    r: float = 3.6
    c: float = 1.0
    cost: float = 1.0
    L: int = 100
    iterations: int = 100000
    seed: int | None = None

    # Fermi-specific
    kappa: float = 0.1  # noise / selection intensity (W = 1/(1+exp((P_i-P_j)/κ)))

    save_frames_interval: int = 1000

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ValueError("L must be > 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")


# ── Engine ──────────────────────────────────────────────────────────────
class FermiSPGGEngine:
    """Synchronous Fermi-rule update on an L×L lattice with periodic boundary."""

    def __init__(self, cfg: FermiSPGGConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # 50/50 random initial strategies: 0=cooperate, 1=defect
        self.actions = self.rng.integers(0, 2, size=(cfg.L, cfg.L), endpoint=False).astype(np.int8)

    # ── Payoff (identical to DQN version) ───────────────────────────────
    def _compute_payoff_raw(self, coop: np.ndarray) -> np.ndarray:
        group_coop_counts = overlap5(coop)
        group_benefit = self.cfg.r * self.cfg.c * group_coop_counts / 5.0
        return overlap5(group_benefit) - 5.0 * self.cfg.cost * coop

    @staticmethod
    def _coop_feature(actions: np.ndarray) -> np.ndarray:
        return (actions == 0).astype(np.float32)

    # ── One Fermi step (vectorised) ─────────────────────────────────────
    def _fermi_step(self) -> None:
        coop = self._coop_feature(self.actions)
        payoff = self._compute_payoff_raw(coop)

        # 4 von-Neumann neighbors via np.roll
        neighbor_strats = [
            np.roll(self.actions, 1, axis=0),   # up
            np.roll(self.actions, -1, axis=0),  # down
            np.roll(self.actions, 1, axis=1),   # left
            np.roll(self.actions, -1, axis=1),  # right
        ]
        neighbor_pays = [
            np.roll(payoff, 1, axis=0),
            np.roll(payoff, -1, axis=0),
            np.roll(payoff, 1, axis=1),
            np.roll(payoff, -1, axis=1),
        ]

        # Each agent picks one random neighbor
        directions = self.rng.integers(0, 4, size=(self.cfg.L, self.cfg.L))
        sel_strat = np.choose(directions, neighbor_strats).astype(np.int8)
        sel_pay = np.choose(directions, neighbor_pays)

        # Fermi adoption probability: W = 1/(1+exp((P_self - P_neighbor)/κ))
        W = 1.0 / (1.0 + np.exp((payoff - sel_pay) / self.cfg.kappa))

        adopt = self.rng.random((self.cfg.L, self.cfg.L)) < W
        self.actions = np.where(adopt, sel_strat, self.actions).astype(np.int8)

    # ── Main loop ───────────────────────────────────────────────────────
    def run(self, output_dir: Path) -> dict[str, float | int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        coop_rate_history: list[float] = []
        avg_payoff_history: list[float] = []
        avg_payoff_norm_history: list[float] = []
        frames: list[np.ndarray] = []

        for step in range(1, self.cfg.iterations + 1):
            self._fermi_step()

            coop = self._coop_feature(self.actions)
            payoff_raw = self._compute_payoff_raw(coop)
            payoff_norm = normalize_payoff_fixed(payoff_raw)

            coop_rate_history.append(float(np.mean(self.actions == 0)))
            avg_payoff_history.append(float(np.mean(payoff_raw)))
            avg_payoff_norm_history.append(float(np.mean(payoff_norm)))

            if self.cfg.save_frames_interval > 0 and step % self.cfg.save_frames_interval == 0:
                frames.append(self.actions.copy())

        # ── Save ────────────────────────────────────────────────────────
        with h5py.File(data_dir / "experiment_data.h5", "w") as f:
            f.create_dataset("config_json", data=np.bytes_(json.dumps(asdict(self.cfg))))
            f.create_dataset("coop_rate_history", data=np.asarray(coop_rate_history, dtype=np.float32))
            f.create_dataset("avg_payoff_history", data=np.asarray(avg_payoff_history, dtype=np.float32))
            f.create_dataset("avg_payoff_norm_history", data=np.asarray(avg_payoff_norm_history, dtype=np.float32))
            f.create_dataset("Sn_final", data=self.actions.astype(np.int8))
            if frames:
                dset = f.create_dataset("Sn_history", data=np.stack(frames, axis=0))
                dset.attrs["save_frames_interval"] = self.cfg.save_frames_interval

        return {
            "final_coop_ratio": float(np.mean(self.actions == 0)),
            "final_def_ratio": float(np.mean(self.actions == 1)),
            "final_avg_payoff": float(avg_payoff_history[-1]) if avg_payoff_history else 0.0,
            "final_avg_payoff_norm": float(avg_payoff_norm_history[-1]) if avg_payoff_norm_history else 0.0,
            "total_steps": self.cfg.iterations,
        }


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fermi-rule SPGG baseline")
    p.add_argument("--output-root", default="results/fermi_spgg")
    p.add_argument("--r", type=float, default=3.6)
    p.add_argument("--grid-size", type=int, default=100)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--save-frames-interval", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FermiSPGGConfig(
        r=args.r,
        L=args.grid_size,
        iterations=args.iterations,
        seed=args.seed,
        kappa=args.kappa,
        save_frames_interval=args.save_frames_interval,
    )
    output_dir = Path(args.output_root) / f"r{cfg.r}_L{cfg.L}_seed{cfg.seed}"
    summary = FermiSPGGEngine(cfg).run(output_dir)
    print(
        f"Done {output_dir} | steps={summary['total_steps']} | "
        f"coop={summary['final_coop_ratio']:.4f} | defect={summary['final_def_ratio']:.4f} | "
        f"avg_payoff={summary['final_avg_payoff']:.4f}"
    )


if __name__ == "__main__":
    main()
