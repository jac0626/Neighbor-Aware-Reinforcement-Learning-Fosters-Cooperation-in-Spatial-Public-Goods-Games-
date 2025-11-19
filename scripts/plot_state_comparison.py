#!/usr/bin/env python3
"""
Script for generating state representation comparison figures
Compares reputation-based vs action-based state representations
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import (
    setup_matplotlib_for_publication,
    plot_state_comparison
)
from src.experiments.runner import get_folder_name


def main():
    parser = argparse.ArgumentParser(description='Generate state comparison figures')
    parser.add_argument('--data-dir', type=str, default='.',
                       help='Base directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='paper_figures',
                       help='Output directory for figures')
    parser.add_argument('--r', type=float, default=4.6,
                       help='r value for comparison')
    parser.add_argument('--kappa', type=float, default=0.0,
                       help='kappa value for comparison')
    parser.add_argument('--use-second-order', action='store_true',
                       help='Use second-order neighbors')
    parser.add_argument('--alpha', type=float, default=0.8,
                       help='Learning rate')
    parser.add_argument('--w-p', type=float, default=1.0,
                       help='Reward weight for payoff')
    parser.add_argument('--rep-gain-c', type=float, default=1.0,
                       help='Reputation gain for cooperation')
    
    args = parser.parse_args()
    
    # Setup plotting style
    setup_matplotlib_for_publication(font_size_pt=12)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_path = args.data_dir
    output_dir = args.output_dir
    
    # Generate folder names for both state representations
    folder_rep = get_folder_name(args.r, args.kappa, args.use_second_order,
                                 args.alpha, args.w_p, args.rep_gain_c, 'reputation')
    folder_action = get_folder_name(args.r, args.kappa, args.use_second_order,
                                   args.alpha, args.w_p, args.rep_gain_c, 'action')
    
    # Define data paths
    comparison_paths = {
        'State: Reputation': os.path.join(base_path, folder_rep, "data", "experiment_data.h5"),
        'State: Previous Action': os.path.join(base_path, folder_action, "data", "experiment_data.h5"),
    }
    
    # Generate comparison figure
    print("\n--- Generating State Comparison Figure ---")
    plot_state_comparison(
        comparison_paths,
        os.path.join(output_dir, "State_Comparison.pdf"),
        total_iterations=100001
    )
    
    print("\nState comparison plotting complete.")


if __name__ == '__main__':
    main()

