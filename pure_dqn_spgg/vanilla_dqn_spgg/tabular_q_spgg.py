"""Tabular Q-learning baseline for Spatial Public Goods Game.

Discretises the same state features used by DQN-local (normalised payoff,
own strategy, local cooperation rate) into a finite Q-table.

State space:  payoff_bin(5) × strategy(2) × neighbor_coop_count(6) = 60 states
Action space: cooperate(0) / defect(1)
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
N_PAYOFF_BINS = 5
N_STRATEGY = 2
N_NEIGHBOR_COOP = 6   # overlap5 of coop gives 0..5
N_STATES = N_PAYOFF_BINS * N_STRATEGY * N_NEIGHBOR_COOP   # 60
N_ACTIONS = 2


@dataclass(slots=True)
class TabularQSPGGConfig:
    # SPGG environment (same defaults as DQN)
    r: float = 3.6
    c: float = 1.0
    cost: float = 1.0
    L: int = 100
    iterations: int = 100000
    seed: int | None = None

    # Q-learning
    alpha: float = 0.1       # learning rate
    gamma: float = 0.9       # discount factor (same as DQN)
    epsilon: float = 0.05    # initial exploration
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.001

    save_frames_interval: int = 1000

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ValueError("L must be > 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")


# ── Engine ──────────────────────────────────────────────────────────────
class TabularQSPGGEngine:
    """Shared Q-table for all L×L agents, updated every step."""

    def __init__(self, cfg: TabularQSPGGConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.actions = self.rng.integers(0, 2, size=(cfg.L, cfg.L), endpoint=False).astype(np.int8)

        # Shared Q-table: (n_states, n_actions) initialised to zeros
        self.q_table = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    # ── Payoff (identical to DQN version) ───────────────────────────────
    def _compute_payoff_raw(self, coop: np.ndarray) -> np.ndarray:
        group_coop_counts = overlap5(coop)
        group_benefit = self.cfg.r * self.cfg.c * group_coop_counts / 5.0
        return overlap5(group_benefit) - 5.0 * self.cfg.cost * coop

    def _normalize_payoff(self, payoff_raw: np.ndarray) -> np.ndarray:
        return normalize_payoff_fixed(payoff_raw)

    @staticmethod
    def _coop_feature(actions: np.ndarray) -> np.ndarray:
        return (actions == 0).astype(np.float32)

    # ── State discretisation ────────────────────────────────────────────
    def _state_indices(self, payoff_norm: np.ndarray, coop: np.ndarray) -> np.ndarray:
        """Map continuous features → integer state index in [0, 59].

        Dimensions:
            payoff_bin      : int(payoff_norm * 5) clamped to [0, 4]   (5 bins)
            self_strategy   : actions ∈ {0, 1}                         (2 values)
            neighbor_coop   : overlap5(coop) ∈ {0, 1, 2, 3, 4, 5}     (6 values)
        """
        payoff_bin = np.clip((payoff_norm * N_PAYOFF_BINS).astype(np.int32), 0, N_PAYOFF_BINS - 1)
        strategy = self.actions.astype(np.int32)
        neighbor_coop = np.clip(overlap5(coop).astype(np.int32), 0, N_NEIGHBOR_COOP - 1)
        return payoff_bin * (N_STRATEGY * N_NEIGHBOR_COOP) + strategy * N_NEIGHBOR_COOP + neighbor_coop

    # ── Action selection ────────────────────────────────────────────────
    def _choose_actions(self, state_idx: np.ndarray, epsilon: float) -> np.ndarray:
        """ε-greedy from shared Q-table."""
        q_vals = self.q_table[state_idx.ravel()]                    # (L*L, 2)
        greedy = np.argmax(q_vals, axis=1).reshape(self.cfg.L, self.cfg.L).astype(np.int8)
        explore = self.rng.random((self.cfg.L, self.cfg.L)) < epsilon
        random_a = self.rng.integers(0, 2, size=(self.cfg.L, self.cfg.L), endpoint=False).astype(np.int8)
        return np.where(explore, random_a, greedy)

    # ── Q-table update (batch-averaged) ─────────────────────────────────
    def _update_q(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        """For each (s, a) pair appearing in the batch, apply the averaged TD error."""
        s_flat = states.ravel()
        a_flat = actions.ravel()
        r_flat = rewards.ravel()
        ns_flat = next_states.ravel()

        max_q_next = np.max(self.q_table[ns_flat], axis=1)          # (L*L,)
        td_target = r_flat + self.cfg.gamma * max_q_next
        current_q = self.q_table[s_flat, a_flat]
        td_error = td_target - current_q

        # Aggregate average TD error per (s, a) pair
        update_sum = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        update_cnt = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        np.add.at(update_sum, (s_flat, a_flat), td_error)
        np.add.at(update_cnt, (s_flat, a_flat), 1.0)

        mask = update_cnt > 0
        self.q_table[mask] += self.cfg.alpha * (update_sum[mask] / update_cnt[mask])

    # ── Main loop ───────────────────────────────────────────────────────
    def run(self, output_dir: Path) -> dict[str, float | int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        coop_rate_history: list[float] = []
        avg_payoff_history: list[float] = []
        avg_payoff_norm_history: list[float] = []
        epsilon_history: list[float] = []
        frames: list[np.ndarray] = []

        epsilon = float(self.cfg.epsilon)

        # Build initial state
        coop = self._coop_feature(self.actions)
        payoff_raw = self._compute_payoff_raw(coop)
        payoff_norm = self._normalize_payoff(payoff_raw)
        state_idx = self._state_indices(payoff_norm, coop)

        for step in range(1, self.cfg.iterations + 1):
            # Choose actions
            next_actions = self._choose_actions(state_idx, epsilon)

            # Environment step
            next_coop = self._coop_feature(next_actions)
            next_payoff_raw = self._compute_payoff_raw(next_coop)
            next_payoff_norm = self._normalize_payoff(next_payoff_raw)
            next_state_idx = self._state_indices(next_payoff_norm, next_coop)

            # Q-table update
            self._update_q(state_idx, next_actions, next_payoff_norm, next_state_idx)

            # Advance
            self.actions = next_actions
            state_idx = next_state_idx
            epsilon = max(epsilon * self.cfg.epsilon_decay, self.cfg.epsilon_min)

            coop_rate_history.append(float(np.mean(self.actions == 0)))
            avg_payoff_history.append(float(np.mean(next_payoff_raw)))
            avg_payoff_norm_history.append(float(np.mean(next_payoff_norm)))
            epsilon_history.append(epsilon)

            if self.cfg.save_frames_interval > 0 and step % self.cfg.save_frames_interval == 0:
                frames.append(self.actions.copy())

        # ── Save ────────────────────────────────────────────────────────
        with h5py.File(data_dir / "experiment_data.h5", "w") as f:
            f.create_dataset("config_json", data=np.bytes_(json.dumps(asdict(self.cfg))))
            f.create_dataset("coop_rate_history", data=np.asarray(coop_rate_history, dtype=np.float32))
            f.create_dataset("avg_payoff_history", data=np.asarray(avg_payoff_history, dtype=np.float32))
            f.create_dataset("avg_payoff_norm_history", data=np.asarray(avg_payoff_norm_history, dtype=np.float32))
            f.create_dataset("epsilon_history", data=np.asarray(epsilon_history, dtype=np.float32))
            f.create_dataset("Sn_final", data=self.actions.astype(np.int8))
            f.create_dataset("q_table", data=self.q_table)
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
    p = argparse.ArgumentParser(description="Tabular Q-learning SPGG baseline")
    p.add_argument("--output-root", default="results/tabular_q_spgg")
    p.add_argument("--r", type=float, default=3.6)
    p.add_argument("--grid-size", type=int, default=100)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--epsilon-min", type=float, default=0.001)
    p.add_argument("--save-frames-interval", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TabularQSPGGConfig(
        r=args.r,
        L=args.grid_size,
        iterations=args.iterations,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        save_frames_interval=args.save_frames_interval,
    )
    output_dir = Path(args.output_root) / f"r{cfg.r}_L{cfg.L}_seed{cfg.seed}"
    summary = TabularQSPGGEngine(cfg).run(output_dir)
    print(
        f"Done {output_dir} | steps={summary['total_steps']} | "
        f"coop={summary['final_coop_ratio']:.4f} | defect={summary['final_def_ratio']:.4f} | "
        f"avg_payoff={summary['final_avg_payoff']:.4f}"
    )


if __name__ == "__main__":
    main()
