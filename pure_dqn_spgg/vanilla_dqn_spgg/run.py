from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from spgg_utils import normalize_payoff_fixed, overlap5, overlap_wide, von_neumann_neighbors_sum


SUPPORTED_STATE_MODES = ("self", "local", "vonn", "vonn_full", "wide", "history")
SUPPORTED_DQN_INIT_MODES = ("torch_default", "zero_last")
SUPPORTED_GREEDY_TIE_BREAKS = ("first", "random")


@dataclass(slots=True)
class VanillaDQNSPGGConfig:
    # SPGG environment
    r: float = 3.6
    c: float = 1.0
    cost: float = 1.0
    L: int = 100
    iterations: int = 100000
    seed: int | None = None
    state_mode: str = "local"
    history_len: int = 3

    # Shared DQN
    action_dim: int = 2
    gamma: float = 0.9
    dqn_lr: float = 1e-4
    hidden_dim: int = 64
    dqn_init_mode: str = "zero_last"
    greedy_tie_break: str = "random"

    # Replay and target network
    buffer_size: int = 2000000
    batch_size: int = 256
    warmup_replay_size: int = 200000
    tau: float = 0.01
    train_steps_per_env_step: int = 4

    # Exploration
    epsilon: float = 0.5
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.0

    # Optional convergence-based stopping
    adaptive_stop: bool = True
    min_steps_before_stop: int = 20000
    stability_window: int = 5000
    stability_check_interval: int = 1000
    stable_checks_required: int = 3
    stability_mean_tol: float = 0.015
    stability_std_tol: float = 0.015
    stability_slope_tol: float = 1e-5

    save_frames_interval: int = 1000
    update_prob: float = 1.0  # Synchronous by default
    save_model: bool = True
    # Runtime
    deterministic_cpu: bool = True

    def __post_init__(self) -> None:
        if self.action_dim != 2:
            raise ValueError("VanillaDQNSPGG uses a fixed binary action space: 0=cooperate, 1=defect")
        if self.L <= 0:
            raise ValueError("L must be > 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.warmup_replay_size < 0:
            raise ValueError("warmup_replay_size must be >= 0")
        if self.tau <= 0.0 or self.tau > 1.0:
            raise ValueError("tau must be in (0, 1]")
        if getattr(self, "train_steps_per_env_step", 1) <= 0:
            raise ValueError("train_steps_per_env_step must be > 0")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")
        if self.epsilon_decay <= 0.0 or self.epsilon_decay > 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if self.epsilon_min < 0.0:
            raise ValueError("epsilon_min must be >= 0")
        if not (0.0 < self.update_prob <= 1.0):
            raise ValueError("update_prob must be in (0, 1]")
        if self.min_steps_before_stop < 0:
            raise ValueError("min_steps_before_stop must be >= 0")
        if self.stability_window <= 1:
            raise ValueError("stability_window must be > 1")
        if self.stability_check_interval <= 0:
            raise ValueError("stability_check_interval must be > 0")
        if self.stable_checks_required <= 0:
            raise ValueError("stable_checks_required must be > 0")
        if self.stability_mean_tol < 0.0:
            raise ValueError("stability_mean_tol must be >= 0")
        if self.stability_std_tol < 0.0:
            raise ValueError("stability_std_tol must be >= 0")
        if self.stability_slope_tol < 0.0:
            raise ValueError("stability_slope_tol must be >= 0")
        if self.state_mode not in SUPPORTED_STATE_MODES:
            raise ValueError(f"Unsupported state_mode: {self.state_mode}")
        if self.history_len <= 0:
            raise ValueError("history_len must be > 0")
        if self.dqn_init_mode not in SUPPORTED_DQN_INIT_MODES:
            raise ValueError(f"Unsupported dqn_init_mode: {self.dqn_init_mode}")
        if self.greedy_tie_break not in SUPPORTED_GREEDY_TIE_BREAKS:
            raise ValueError(f"Unsupported greedy_tie_break: {self.greedy_tie_break}")

    def state_dim(self) -> int:
        if self.state_mode == "self":
            return 2
        if self.state_mode == "local":
            return 3
        if self.state_mode == "vonn_full":
            return 6  # [P, A, N1, N2, N3, N4]
        if self.state_mode == "vonn":
            return 3  # [P, A, von_neumann_density]
        if self.state_mode == "wide":
            return 4  # local + wide neighbor features
        if self.state_mode == "history":
            return 3 * self.history_len
        raise ValueError(f"Unsupported state_mode: {self.state_mode}")


def run_dir_name(cfg: VanillaDQNSPGGConfig) -> str:
    tags = ["vanilla", "p1", "dqn", "tgthard", "basic", f"st-{cfg.state_mode}"]
    tags.append(f"init-{cfg.dqn_init_mode}")
    tags.append(f"tie-{cfg.greedy_tie_break}")
    if cfg.deterministic_cpu:
        tags.append("dcpu")
    if cfg.adaptive_stop:
        tags.append("astop")

    return (
        f"r{cfg.r}_L{cfg.L}_it{cfg.iterations}"
        f"_eps{cfg.epsilon}_ed{cfg.epsilon_decay}_emin{cfg.epsilon_min}"
        f"_lr{cfg.dqn_lr}_bs{cfg.batch_size}_wr{cfg.warmup_replay_size}"
        f"_seed{cfg.seed if cfg.seed is not None else 'none'}"
        f"_feat{'-'.join(tags)}"
    )


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = max(1, int(capacity))
        self.state_dim = int(state_dim)
        self.position = 0
        self.size = 0

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool,
    ) -> None:
        s = states.reshape(-1, self.state_dim).astype(np.float32, copy=False)
        a = actions.reshape(-1).astype(np.int64, copy=False)
        r = rewards.reshape(-1).astype(np.float32, copy=False)
        ns = next_states.reshape(-1, self.state_dim).astype(np.float32, copy=False)
        d = np.full(s.shape[0], 1.0 if done else 0.0, dtype=np.float32)

        n = s.shape[0]
        idx = (np.arange(n, dtype=np.int64) + self.position) % self.capacity
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = ns
        self.dones[idx] = d

        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, ...]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        idx = rng.integers(0, self.size, size=int(batch_size), endpoint=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class DQNNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedDQN:
    def __init__(self, cfg: VanillaDQNSPGGConfig, rng: np.random.Generator, state_dim: int):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install it with: pip install torch")

        torch_seed = int(rng.integers(0, 2**31))
        torch.manual_seed(torch_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        if cfg.deterministic_cpu:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.gamma = float(cfg.gamma)
        self.state_dim = int(state_dim)
        self.online_net = DQNNet(self.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self._apply_initialization(self.online_net, cfg.dqn_init_mode)
        self.target_net = DQNNet(self.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=cfg.dqn_lr)

    @staticmethod
    def _apply_initialization(net: DQNNet, mode: str) -> None:
        if mode == "torch_default":
            return
        if mode == "zero_last":
            final_layer = net.net[2]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
            return
        raise ValueError(f"Unsupported dqn_init_mode: {mode}")

    @torch.no_grad()
    def predict(self, states: np.ndarray) -> np.ndarray:
        self.online_net.eval()
        s = torch.from_numpy(states.astype(np.float32)).to(self.device)
        return self.online_net(s).cpu().numpy()

    def train_step(self, batch: tuple[np.ndarray, ...]) -> float:
        states, actions, rewards, next_states, dones = batch
        s = torch.from_numpy(states).to(self.device)
        a = torch.from_numpy(actions).long().to(self.device)
        r = torch.from_numpy(rewards).to(self.device)
        ns = torch.from_numpy(next_states).to(self.device)
        d = torch.from_numpy(dones).to(self.device)

        self.online_net.train()
        q_current = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_q_next = self.target_net(ns).max(dim=1)[0]
            td_target = r + (1.0 - d) * self.gamma * max_q_next

        loss = F.smooth_l1_loss(q_current, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self, tau: float) -> None:
        for t_param, o_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)


class VanillaDQNSPGGEngine:
    """Vanilla shared-parameter DQN for spatial public goods games."""

    def __init__(self, cfg: VanillaDQNSPGGConfig):
        self.cfg = cfg
        self.state_dim = cfg.state_dim()
        self.state_history: deque[np.ndarray] = deque(maxlen=cfg.history_len)
        self.rng = np.random.default_rng(cfg.seed)
        self.actions = self.rng.integers(0, 2, size=(cfg.L, cfg.L), endpoint=False).astype(np.int8)

        self.replay_buffer = ReplayBuffer(cfg.buffer_size, self.state_dim)
        self.dqn = SharedDQN(cfg, self.rng, self.state_dim)

    def _compute_payoff_raw(self, coop: np.ndarray) -> np.ndarray:
        # Each agent receives benefit from 5 overlapping groups and pays cooperation cost in all 5.
        group_coop_counts = overlap5(coop)
        group_benefit = self.cfg.r * self.cfg.c * group_coop_counts / 5.0
        return overlap5(group_benefit) - 5.0 * self.cfg.cost * coop

    def _normalize_payoff(self, payoff_raw: np.ndarray) -> np.ndarray:
        return normalize_payoff_fixed(payoff_raw)

    @staticmethod
    def _coop_feature(actions: np.ndarray) -> np.ndarray:
        return (actions == 0).astype(np.float32)

    def _build_state(self, payoff_norm: np.ndarray, coop: np.ndarray) -> np.ndarray:
        if self.cfg.state_mode == "self":
            return np.stack([payoff_norm, coop], axis=-1).astype(np.float32)
            
        elif self.cfg.state_mode == "vonn":
            density = von_neumann_neighbors_sum(coop) / 4.0
            return np.stack([payoff_norm, coop, density], axis=-1).astype(np.float32)
            
        elif self.cfg.state_mode in ["local", "history"]:
            local_neighbors = overlap5(coop) / 5.0
            st = np.stack([payoff_norm, coop, local_neighbors], axis=-1).astype(np.float32)
            
            if self.cfg.state_mode == "history":
                self.state_history.append(st)
                # Left-pad with zeros if history is not yet full
                padded = list(self.state_history)
                while len(padded) < self.cfg.history_len:
                    padded.insert(0, np.zeros_like(st))
                return np.concatenate(padded, axis=-1).astype(np.float32)
            return st
            
        elif self.cfg.state_mode == "vonn_full":
            # Explicit actions of 4 neighbors: up, down, left, right
            a = coop
            n1 = np.roll(a, 1, axis=0) # up
            n2 = np.roll(a, -1, axis=0) # down
            n3 = np.roll(a, 1, axis=1) # left
            n4 = np.roll(a, -1, axis=1) # right
            return np.stack([payoff_norm, a, n1, n2, n3, n4], axis=-1).astype(np.float32)
            
        elif self.cfg.state_mode == "wide":
            local_neighbors = overlap5(coop) / 5.0
            wide_neighbors = overlap_wide(coop) / 25.0
            return np.stack([payoff_norm, coop, local_neighbors, wide_neighbors], axis=-1).astype(np.float32)
        
        return np.stack([payoff_norm], axis=-1).astype(np.float32)

    def _is_converged(self, coop_rate_history: list[float]) -> bool:
        w = self.cfg.stability_window
        if len(coop_rate_history) < 2 * w:
            return False

        hist = np.asarray(coop_rate_history, dtype=np.float64)
        prev = hist[-2 * w : -w]
        tail = hist[-w:]

        mean_shift = abs(float(np.mean(tail) - np.mean(prev)))
        std_shift = abs(float(np.std(tail, ddof=0) - np.std(prev, ddof=0)))
        x = np.arange(w, dtype=np.float64)
        slope = float(np.polyfit(x, tail, 1)[0])

        return (
            mean_shift <= self.cfg.stability_mean_tol
            and std_shift <= self.cfg.stability_std_tol
            and abs(slope) <= self.cfg.stability_slope_tol
        )

    def _choose_actions(self, state: np.ndarray, epsilon: float, current_actions: np.ndarray) -> np.ndarray:
        # Decide who is allowed to change their action this step
        update_mask = self.rng.random((self.cfg.L, self.cfg.L)) < self.cfg.update_prob
        
        # Calculate greedy actions for everyone
        flat_state = state.reshape(-1, self.state_dim)
        q_values = self.dqn.predict(flat_state).reshape(self.cfg.L, self.cfg.L, self.cfg.action_dim)
        greedy_actions = self._greedy_actions(q_values)
        
        # Calculate random exploration actions
        explore = self.rng.random((self.cfg.L, self.cfg.L)) < epsilon
        random_actions = self.rng.integers(0, 2, size=(self.cfg.L, self.cfg.L), endpoint=False).astype(np.int8)
        
        # New proposed actions (epsilon-greedy)
        new_proposed = np.where(explore, random_actions, greedy_actions)
        
        # Only apply new actions where mask is true, otherwise stay put
        return np.where(update_mask, new_proposed, current_actions).astype(np.int8)

    def _greedy_actions(self, q_values: np.ndarray) -> np.ndarray:
        if self.cfg.greedy_tie_break == "first":
            return np.argmax(q_values, axis=2).astype(np.int8)

        max_q = np.max(q_values, axis=2, keepdims=True)
        tie_mask = np.isclose(q_values, max_q, rtol=1e-7, atol=1e-8)
        random_scores = np.where(tie_mask, self.rng.random(q_values.shape), -1.0)
        return np.argmax(random_scores, axis=2).astype(np.int8)

    def run(self, output_dir: Path) -> dict[str, float | int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        models_dir = output_dir / "models"
        if self.cfg.save_model:
            models_dir.mkdir(parents=True, exist_ok=True)

        coop_rate_history: list[float] = []
        avg_payoff_history: list[float] = []
        avg_payoff_norm_history: list[float] = []
        epsilon_history: list[float] = []
        loss_history: list[float] = []
        frames: list[np.ndarray] = []

        epsilon = self.cfg.epsilon
        stable_hits = 0

        initial_coop = self._coop_feature(self.actions)
        initial_payoff_raw = self._compute_payoff_raw(initial_coop)
        initial_payoff_norm = self._normalize_payoff(initial_payoff_raw)
        state = self._build_state(initial_payoff_norm, initial_coop)

        for step in range(1, self.cfg.iterations + 1):
            if self.replay_buffer.size < self.cfg.warmup_replay_size:
                next_actions = self.rng.integers(0, 2, size=(self.cfg.L, self.cfg.L), endpoint=False).astype(np.int8)
            else:
                next_actions = self._choose_actions(state, epsilon, self.actions)

            next_coop = self._coop_feature(next_actions)
            next_payoff_raw = self._compute_payoff_raw(next_coop)
            next_payoff_norm = self._normalize_payoff(next_payoff_raw)
            next_state = self._build_state(next_payoff_norm, next_coop)

            self.replay_buffer.add_batch(
                state.reshape(-1, self.state_dim),
                next_actions.reshape(-1),
                next_payoff_norm.reshape(-1),
                next_state.reshape(-1, self.state_dim),
                done=False,
            )
            
            self.actions = next_actions
            state = next_state
            
            coop_rate_history.append(float(np.mean(self.actions == 0)))
            avg_payoff_history.append(float(np.mean(next_payoff_raw)))
            avg_payoff_norm_history.append(float(np.mean(next_payoff_norm)))
            epsilon_history.append(float(epsilon))

            if self.replay_buffer.size >= max(self.cfg.warmup_replay_size, self.cfg.batch_size):
                avg_loss = 0.0
                for _ in range(self.cfg.train_steps_per_env_step):
                    batch = self.replay_buffer.sample(self.rng, self.cfg.batch_size)
                    avg_loss += self.dqn.train_step(batch)
                loss_history.append(avg_loss / self.cfg.train_steps_per_env_step)
                self.dqn.sync_target(self.cfg.tau)
                epsilon = max(epsilon * self.cfg.epsilon_decay, self.cfg.epsilon_min)
            else:
                loss_history.append(0.0)

            self.actions = next_actions
            
            if self.cfg.save_frames_interval > 0 and step % self.cfg.save_frames_interval == 0:
                frames.append(self.actions.copy())
                
            if (
                self.cfg.adaptive_stop
                and step >= self.cfg.min_steps_before_stop
                and step % self.cfg.stability_check_interval == 0
            ):
                if self._is_converged(coop_rate_history):
                    stable_hits += 1
                else:
                    stable_hits = 0

                if stable_hits >= self.cfg.stable_checks_required:
                    break

        with h5py.File(data_dir / "experiment_data.h5", "w") as f:
            f.create_dataset("config_json", data=np.bytes_(json.dumps(asdict(self.cfg))))
            f.create_dataset("coop_rate_history", data=np.asarray(coop_rate_history, dtype=np.float32))
            f.create_dataset("avg_payoff_history", data=np.asarray(avg_payoff_history, dtype=np.float32))
            f.create_dataset("avg_payoff_norm_history", data=np.asarray(avg_payoff_norm_history, dtype=np.float32))
            f.create_dataset("epsilon_history", data=np.asarray(epsilon_history, dtype=np.float32))
            f.create_dataset("loss_history", data=np.asarray(loss_history, dtype=np.float32))
            f.create_dataset("Sn_final", data=self.actions.astype(np.int8))
            if len(frames) > 0:
                dset = f.create_dataset("Sn_history", data=np.stack(frames, axis=0))
                dset.attrs['save_frames_interval'] = self.cfg.save_frames_interval

        if self.cfg.save_model:
            torch.save(
                {
                    "online_net_state": self.dqn.online_net.state_dict(),
                    "target_net_state": self.dqn.target_net.state_dict(),
                    "device": str(self.dqn.device),
                    "state_dim": self.state_dim,
                    "action_dim": self.cfg.action_dim,
                },
                models_dir / "dqn.pt",
            )

        return {
            "final_coop_ratio": coop_rate_history[-1],
            "final_def_ratio": 1.0 - coop_rate_history[-1],
            "final_avg_payoff": avg_payoff_history[-1] if avg_payoff_history else 0.0,
            "final_avg_payoff_norm": avg_payoff_norm_history[-1] if avg_payoff_norm_history else 0.0,
            "total_steps": step,
            "coop_rate_history": coop_rate_history,
            "avg_payoff_history": avg_payoff_history,
            "avg_payoff_norm_history": avg_payoff_norm_history,
            "epsilon_history": epsilon_history,
            "loss_history": loss_history,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-file vanilla DQN SPGG runner")
    p.add_argument("--output-root", default="results/vanilla_dqn_spgg")
    p.add_argument("--r", type=float, default=3.6)
    p.add_argument("--grid-size", type=int, default=100)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--state-mode", type=str, default="local", choices=list(SUPPORTED_STATE_MODES))
    p.add_argument("--history-len", type=int, default=3)

    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--dqn-lr", type=float, default=1e-4) # Reduced further to damp oscillation
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dqn-init-mode", type=str, default="zero_last", choices=list(SUPPORTED_DQN_INIT_MODES))
    p.add_argument("--greedy-tie-break", type=str, default="random", choices=list(SUPPORTED_GREEDY_TIE_BREAKS))
    p.add_argument("--buffer-size", type=int, default=2000000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-replay-size", type=int, default=200000)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--train-steps-per-env-step", type=int, default=4)
    p.add_argument("--update-prob", type=float, default=1.0, help="Prob an agent updates its action")

    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--epsilon-min", type=float, default=0.0)

    p.add_argument("--disable-adaptive-stop", action="store_true")
    p.add_argument("--min-steps-before-stop", type=int, default=3000)
    p.add_argument("--stability-window", type=int, default=500)
    p.add_argument("--stability-check-interval", type=int, default=50)
    p.add_argument("--stable-checks-required", type=int, default=3)
    p.add_argument("--stability-mean-tol", type=float, default=0.01)
    p.add_argument("--stability-std-tol", type=float, default=0.02)
    p.add_argument("--stability-slope-tol", type=float, default=1e-4)

    p.add_argument("--disable-deterministic-cpu", action="store_true")
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--save-frames-interval", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = VanillaDQNSPGGConfig(
        r=args.r,
        L=args.grid_size,
        iterations=args.iterations,
        seed=args.seed,
        state_mode=args.state_mode,
        history_len=args.history_len,
        gamma=args.gamma,
        dqn_lr=args.dqn_lr,
        hidden_dim=args.hidden_dim,
        dqn_init_mode=args.dqn_init_mode,
        greedy_tie_break=args.greedy_tie_break,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        warmup_replay_size=args.warmup_replay_size,
        tau=args.tau,
        train_steps_per_env_step=args.train_steps_per_env_step,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        update_prob=args.update_prob,
        adaptive_stop=not args.disable_adaptive_stop,
        min_steps_before_stop=args.min_steps_before_stop,
        stability_window=args.stability_window,
        stability_check_interval=args.stability_check_interval,
        stable_checks_required=args.stable_checks_required,
        stability_mean_tol=args.stability_mean_tol,
        stability_std_tol=args.stability_std_tol,
        stability_slope_tol=args.stability_slope_tol,
        save_model=not args.no_save_model,
        save_frames_interval=args.save_frames_interval,
        deterministic_cpu=not args.disable_deterministic_cpu,
    )

    run_dir = output_root / run_dir_name(cfg)
    summary = VanillaDQNSPGGEngine(cfg).run(run_dir)
    print(
        f"Done {run_dir} | steps={summary['total_steps']} | "
        f"coop={summary['final_coop_ratio']:.4f} | defect={summary['final_def_ratio']:.4f} | "
        f"avg_payoff={summary['final_avg_payoff']:.4f}"
    )


if __name__ == "__main__":
    main()
