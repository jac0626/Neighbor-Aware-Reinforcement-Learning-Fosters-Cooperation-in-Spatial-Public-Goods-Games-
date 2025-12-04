#!/usr/bin/env python3
"""
Baseline Comparison Experiments for Dual-Brain Paper.

This script runs baseline methods for comparison:
1. Q-learning only (λ=0, no DQN) - Already supported via dqn_lambda=0
2. DQN only (λ=1, no Q-table) - Already supported via dqn_lambda=1
3. Fermi update - Traditional game theory approach
4. Imitation learning - Copy best neighbor strategy
5. Dual-brain (λ=0.6) - Our proposed method
"""
import argparse
import os
import sys
import numpy as np
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import SimulationConfig
from src.core.spgg_model import SPGG
from src.core.state_strategies import ReputationStateProvider


# Calculate 5-neighborhood overlap
overlap5 = lambda A: A + np.roll(A, -1, 0) + np.roll(A, 1, 0) + np.roll(A, -1, 1) + np.roll(A, 1, 1)


class FermiSPGG:
    """
    Classic Spatial Public Goods Game with Fermi update rule.
    
    Two-strategy version:
    - Strategy 0: Cooperator
    - Strategy 1: Defector
    
    Payoff calculation (per 5-member group):
    - Cooperator: r * c * Nc / 5 - cost
    - Defector: r * c * Nc / 5
    
    Each agent participates in 5 groups, total payoff = sum of all 5 groups.
    """
    
    def __init__(self, config, K=0.1):
        self.config = config
        self.L = config.L
        self.K = K  # Temperature parameter
        self.r = config.r
        self.c = config.c
        self.cost = config.cost
        
        # Initialize: random 50% cooperators, 50% defectors
        # 0 = Cooperator, 1 = Defector
        self._Sn = np.random.randint(0, 2, size=(self.L, self.L))
        self._update_S()
        
    def _update_S(self):
        """Update strategy matrices."""
        self._S = [(self._Sn == j).astype(int) for j in range(2)]
    
    def N(self, group_offset=(0, 0)):
        """Count number of each strategy in each group."""
        S = self._S
        if group_offset != (0, 0):
            S = [np.roll(s, group_offset, axis=(0, 1)) for s in S]
        return [overlap5(s) for s in S]
    
    def P_g_m(self, group_offset=(0, 0), member_offset=(0, 0)):
        """Calculate payoff for a single group."""
        N = self.N(group_offset)
        S = self._S
        if group_offset != (0, 0):
            S = [np.roll(s, group_offset, axis=(0, 1)) for s in S]
        if member_offset != (0, 0):
            S = [np.roll(s, member_offset, axis=(0, 1)) for s in S]
        
        r, c, cost = self.r, self.c, self.cost
        n = 5
        Nc = N[0]  # Number of cooperators
        S_coop, S_defect = S[0], S[1]
        
        # Payoff: Cooperator: r*c*Nc/n - cost, Defector: r*c*Nc/n
        P = (r * c * Nc / n - cost) * S_coop + (r * c * Nc / n) * S_defect
        return P
    
    def compute_total_payoff(self):
        """Compute total payoff (sum of all 5 groups)."""
        P = (self.P_g_m() + 
             self.P_g_m((1, 0), (-1, 0)) + 
             self.P_g_m((-1, 0), (1, 0)) + 
             self.P_g_m((0, 1), (0, -1)) + 
             self.P_g_m((0, -1), (0, 1)))
        return P
    
    def fermi_update(self, P):
        """Vectorized Fermi update rule."""
        L, K = self.L, self.K
        S_in_one = self._Sn
        
        # Compute Fermi probability for each direction
        W_w = 1 / (1 + np.exp((P - np.roll(P, 1, 1)) / K))
        W_e = 1 / (1 + np.exp((P - np.roll(P, -1, 1)) / K))
        W_n = 1 / (1 + np.exp((P - np.roll(P, 1, 0)) / K))
        W_s = 1 / (1 + np.exp((P - np.roll(P, -1, 0)) / K))
        
        # Randomly select neighbor
        RandomNeighbour = np.random.randint(0, 4, size=(L, L))
        Random01 = np.random.uniform(0, 1, size=(L, L))
        
        # Decide whether to adopt neighbor's strategy based on Fermi probability
        S_new = ((RandomNeighbour == 0) * ((Random01 <= W_w) * np.roll(S_in_one, 1, 1) + (Random01 > W_w) * S_in_one) +
                 (RandomNeighbour == 1) * ((Random01 <= W_e) * np.roll(S_in_one, -1, 1) + (Random01 > W_e) * S_in_one) +
                 (RandomNeighbour == 2) * ((Random01 <= W_n) * np.roll(S_in_one, 1, 0) + (Random01 > W_n) * S_in_one) +
                 (RandomNeighbour == 3) * ((Random01 <= W_s) * np.roll(S_in_one, -1, 0) + (Random01 > W_s) * S_in_one))
        
        self._Sn = S_new.astype(int)
        self._update_S()
    
    def run(self, iterations):
        """Run simulation."""
        coop_history = []
        
        for i in range(iterations):
            coop_rate = np.sum(self._S[0]) / (self.L * self.L)
            coop_history.append(coop_rate)
            
            P = self.compute_total_payoff()
            self.fermi_update(P)
            
            if coop_rate <= 0.001 or coop_rate >= 0.999:
                coop_history.extend([coop_rate] * (iterations - i - 1))
                break
        
        return np.array(coop_history)


class ImitationSPGG:
    """
    SPGG with Imitation Learning (copy best neighbor).
    
    Each agent copies the strategy of their highest-payoff neighbor.
    """
    def __init__(self, config):
        self.config = config
        self.L = config.L
        
        # Initialize strategies randomly
        self._Sn = np.random.randint(0, 2, size=(self.L, self.L))
        self._S = [(self._Sn == j).astype(int) for j in range(2)]
        
        self.R = np.zeros((self.L, self.L))
        self.cache = {}
        
    def S(self, group_offset=(0, 0), member_offset=(0, 0)):
        result = self._S
        if group_offset != (0, 0):
            result = [np.roll(s, group_offset, axis=(0, 1)) for s in result]
        if member_offset != (0, 0):
            result = [np.roll(s, member_offset, axis=(0, 1)) for s in result]
        return result
    
    def compute_payoff(self):
        """Compute payoff for each agent."""
        L = self.L
        S = self.S()
        
        n_coop = S[0].copy()
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            n_coop += np.roll(S[0], (di, dj), axis=(0, 1))
        
        r = self.config.r
        c = self.config.c
        cost = self.config.cost
        n = 5
        
        payoff_coop = r * c * n_coop / n - cost
        payoff_defect = r * c * n_coop / n
        
        P = np.where(self._Sn == 0, payoff_coop, payoff_defect)
        return P
    
    def imitation_update(self, P):
        """Copy strategy of best neighbor."""
        L = self.L
        new_Sn = self._Sn.copy()
        
        # For each agent, find best among self and neighbors
        neighbor_offsets = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for i in range(L):
            for j in range(L):
                best_payoff = P[i, j]
                best_strategy = self._Sn[i, j]
                
                for di, dj in neighbor_offsets:
                    ni, nj = (i + di) % L, (j + dj) % L
                    if P[ni, nj] > best_payoff:
                        best_payoff = P[ni, nj]
                        best_strategy = self._Sn[ni, nj]
                
                new_Sn[i, j] = best_strategy
        
        self._Sn = new_Sn
        self._S = [(new_Sn == j).astype(int) for j in range(2)]
        self.cache = {}
    
    def run(self, iterations):
        """Run Imitation simulation."""
        coop_history = []
        
        for i in range(iterations):
            coop_rate = np.sum(self._Sn == 0) / (self.L * self.L)
            coop_history.append(coop_rate)
            
            P = self.compute_payoff()
            self.imitation_update(P)
            
            if coop_rate == 0 or coop_rate == 1:
                break
        
        return np.array(coop_history)


def run_baseline_experiment(method, config, iterations, output_dir):
    """Run a baseline experiment and save results."""
    import h5py
    
    print(f"Running {method} baseline with r={config.r}...")
    
    if method == 'q_only':
        # Q-learning only (λ=0)
        config.use_dqn = True
        config.dqn_lambda = 0.0
        state_provider = ReputationStateProvider(config)
        folder = os.path.join(output_dir, f"baseline_{method}_r{config.r}")
        model = SPGG(config, state_provider, folder=folder)
        h5_path = os.path.join(folder, "data", "experiment_data.h5")
        model.run(h5_path)
        return
    
    elif method == 'dqn_only':
        # DQN only (λ=1)
        config.use_dqn = True
        config.dqn_lambda = 1.0
        state_provider = ReputationStateProvider(config)
        folder = os.path.join(output_dir, f"baseline_{method}_r{config.r}")
        model = SPGG(config, state_provider, folder=folder)
        h5_path = os.path.join(folder, "data", "experiment_data.h5")
        model.run(h5_path)
        return
    
    elif method == 'dual_brain':
        # Dual-brain (λ=0.6)
        config.use_dqn = True
        config.dqn_lambda = 0.6
        state_provider = ReputationStateProvider(config)
        folder = os.path.join(output_dir, f"baseline_{method}_r{config.r}")
        model = SPGG(config, state_provider, folder=folder)
        h5_path = os.path.join(folder, "data", "experiment_data.h5")
        model.run(h5_path)
        return
    
    elif method == 'fermi':
        model = FermiSPGG(config)
        coop_history = model.run(iterations)
    
    elif method == 'imitation':
        model = ImitationSPGG(config)
        coop_history = model.run(iterations)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save results for non-SPGG methods
    result_dir = os.path.join(output_dir, f"baseline_{method}_r{config.r}", "data")
    os.makedirs(result_dir, exist_ok=True)
    
    with h5py.File(os.path.join(result_dir, "experiment_data.h5"), 'w') as f:
        f.create_dataset('coop_rate_history', data=coop_history)
        f.attrs['method'] = method
        f.attrs['r'] = config.r
        f.attrs['L'] = config.L
        f.attrs['iterations'] = len(coop_history)
    
    print(f"Saved {method} results: final coop = {coop_history[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Run baseline comparison experiments')
    parser.add_argument('--method', type=str, required=True,
                       choices=['q_only', 'dqn_only', 'fermi', 'imitation', 'dual_brain'],
                       help='Baseline method to run')
    parser.add_argument('--r', type=float, default=4.0, help='Synergy factor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--iterations', type=int, default=100000, help='Number of iterations')
    parser.add_argument('--L', type=int, default=50, help='Grid size')
    parser.add_argument('--output_dir', type=str, default='baseline_results/')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create config
    config = SimulationConfig(
        L=args.L,
        r=args.r,
        iterations=args.iterations,
        influence_factor=0.0,
        use_dqn=True,
        dqn_tau=0.005,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_baseline_experiment(args.method, config, args.iterations, args.output_dir)


if __name__ == "__main__":
    main()
