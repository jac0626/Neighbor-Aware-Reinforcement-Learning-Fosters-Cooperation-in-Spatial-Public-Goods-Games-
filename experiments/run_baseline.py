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


class FermiSPGG:
    """
    SPGG with Fermi update rule (traditional game theory).
    
    Each agent copies a random neighbor's strategy with probability:
    P = 1 / (1 + exp((P_self - P_neighbor) / K))
    """
    def __init__(self, config):
        self.config = config
        self.L = config.L
        self.K = config.K  # Temperature parameter
        
        # Initialize strategies randomly (0=Cooperate, 1=Defect)
        self._Sn = np.random.randint(0, 2, size=(self.L, self.L))
        self._S = [(self._Sn == j).astype(int) for j in range(2)]
        
        # Reputation (not used in Fermi, but kept for compatibility)
        self.R = np.zeros((self.L, self.L))
        
        self.cache = {}
        
    def S(self, group_offset=(0, 0), member_offset=(0, 0)):
        result = self._S
        if group_offset != (0, 0):
            result = [np.roll(s, group_offset, axis=(0, 1)) for s in result]
        if member_offset != (0, 0):
            result = [np.roll(s, member_offset, axis=(0, 1)) for s in result]
        return result
    
    def N(self, group_offset=(0, 0), member_offset=(0, 0)):
        S = self.S(group_offset, member_offset)
        return [np.sum(S, axis=0) for s in S]
    
    def compute_payoff(self):
        """Compute payoff for each agent."""
        L = self.L
        S = self.S()
        
        # Number of cooperators in each 5-agent group (self + 4 neighbors)
        n_coop = S[0].copy()
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            n_coop += np.roll(S[0], (di, dj), axis=(0, 1))
        
        # Payoff: cooperators pay cost, but get share of public goods
        r = self.config.r
        c = self.config.c
        cost = self.config.cost
        n = 5  # group size
        
        payoff_coop = r * c * n_coop / n - cost
        payoff_defect = r * c * n_coop / n
        
        P = np.where(self._Sn == 0, payoff_coop, payoff_defect)
        return P
    
    def fermi_update(self, P):
        """Apply Fermi update rule."""
        L = self.L
        new_Sn = self._Sn.copy()
        
        # For each agent, pick a random neighbor
        neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for i in range(L):
            for j in range(L):
                # Pick random neighbor
                di, dj = random.choice(neighbor_offsets)
                ni, nj = (i + di) % L, (j + dj) % L
                
                # Fermi probability
                p_self = P[i, j]
                p_neighbor = P[ni, nj]
                prob = 1.0 / (1.0 + np.exp((p_self - p_neighbor) / self.K))
                
                # Copy neighbor's strategy with probability prob
                if np.random.rand() < prob:
                    new_Sn[i, j] = self._Sn[ni, nj]
        
        self._Sn = new_Sn
        self._S = [(new_Sn == j).astype(int) for j in range(2)]
        self.cache = {}
    
    def run(self, iterations):
        """Run Fermi simulation."""
        coop_history = []
        
        for i in range(iterations):
            # Record cooperation rate
            coop_rate = np.sum(self._Sn == 0) / (self.L * self.L)
            coop_history.append(coop_rate)
            
            # Compute payoff
            P = self.compute_payoff()
            
            # Fermi update
            self.fermi_update(P)
            
            # Early stopping
            if coop_rate == 0 or coop_rate == 1:
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
        model = SPGG(config)
        model.run(os.path.join(output_dir, f"baseline_{method}_r{config.r}"))
        return
    
    elif method == 'dqn_only':
        # DQN only (λ=1)
        config.use_dqn = True
        config.dqn_lambda = 1.0
        model = SPGG(config)
        model.run(os.path.join(output_dir, f"baseline_{method}_r{config.r}"))
        return
    
    elif method == 'dual_brain':
        # Dual-brain (λ=0.6)
        config.use_dqn = True
        config.dqn_lambda = 0.6
        model = SPGG(config)
        model.run(os.path.join(output_dir, f"baseline_{method}_r{config.r}"))
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
