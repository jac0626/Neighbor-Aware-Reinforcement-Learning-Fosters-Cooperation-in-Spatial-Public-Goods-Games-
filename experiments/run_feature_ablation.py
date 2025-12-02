#!/usr/bin/env python3
"""
Feature ablation experiment runner.
Tests different DQN input feature combinations to validate their importance.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimulationConfig
from src.core.spgg_model import SPGG
from src.core.state_strategies import ReputationStateProvider

# Feature set mappings
FEATURE_SETS = {
    'full': None,  # Use all default features
    'no_rep': 'no_reputation',
    'no_neighbor': 'no_neighbor_info',
    'only_payoff': 'payoff_only',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_set', required=True, choices=FEATURE_SETS.keys())
    parser.add_argument('--r', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--output_dir', default='exp2_results')
    args = parser.parse_args()
    
    # Base configuration
    config_dict = {
        'L': 100,
        'r': args.r,
        'c': 1.0,
        'cost': 1.0,
        'iterations': args.iterations,
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon': 0.5,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'influence_factor': 0.0,  # Disabled
        'use_dqn': True,
        'dqn_lambda': 0.6,  # Hybrid architecture
        'use_soft_update': True,
        'dqn_tau': 0.005,
        'dqn_feature_set': FEATURE_SETS[args.feature_set],
    }
    
    config = SimulationConfig.from_dict(config_dict)
    
    # Create output folder
    folder_name = f"r{args.r}_{args.feature_set}_seed{args.seed}"
    folder = os.path.join(args.output_dir, folder_name)
    os.makedirs(os.path.join(folder, "data"), exist_ok=True)
    
    # Run simulation
    state_provider = ReputationStateProvider(config)
    spgg = SPGG(config, state_provider, folder=folder)
    filename = os.path.join(folder, "data", "experiment_data.h5")
    
    print(f"Running feature ablation: {args.feature_set}, r={args.r}, seed={args.seed}")
    results = spgg.run(filename)
    
    print(f"Final cooperation rate: {results[0]:.2%}")
    print(f"Results saved to: {filename}")

if __name__ == "__main__":
    main()
