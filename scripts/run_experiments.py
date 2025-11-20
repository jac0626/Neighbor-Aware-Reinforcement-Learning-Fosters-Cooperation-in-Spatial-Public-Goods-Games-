#!/usr/bin/env python3
"""
Main script for running SPGG experiments
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config, generate_param_combinations
from src.experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description='Run SPGG experiments')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: use default config)')
    parser.add_argument('--experiment-type', type=str, 
                       choices=['figure_2_3_4', 'figure_6_7_8_9', 'custom'],
                       default='custom',
                       help='Type of experiment to run')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of processes to use (default: CPU count)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Generate parameter combinations
    print(f"Generating parameter combinations for '{args.experiment_type}'...")
    param_combinations = generate_param_combinations(config, args.experiment_type)
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    print("Parameters to run:")
    for params in param_combinations[:10]:  # Show first 10
        if len(params) == 7:
            r, kappa, use_so, alpha, w_P, rep_gain_C, state_rep = params
        elif len(params) == 6:
            r, kappa, use_so, alpha, w_P, rep_gain_C = params
            state_rep = None
        else:
            raise ValueError(f"Unexpected parameter format: {params}")
        rep_info = f", state={state_rep}" if state_rep is not None else ""
        print(f"  - r={r}, κ={kappa}, M={2 if use_so else 1}, α={alpha}, w_P={w_P}, ΔR_C={rep_gain_C}{rep_info}")
    if len(param_combinations) > 10:
        print(f"  ... and {len(param_combinations) - 10} more")
    
    # Run experiments
    print("\nStarting experiments...")
    results = run_experiments(
        param_combinations,
        num_processes=args.num_processes,
        use_progress_bar=not args.no_progress
    )
    
    print(f"\nCompleted {len(results)} experiments")
    print("Results saved to respective result directories")


if __name__ == '__main__':
    main()

