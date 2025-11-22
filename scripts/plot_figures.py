#!/usr/bin/env python3
"""
Script for generating publication figures from experiment data
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import (
    setup_matplotlib_for_publication,
    plot_figure_2, plot_figure_4, plot_figure_6,
    plot_figure_7, plot_figure_8, plot_figure_9
)
from src.experiments.runner import get_folder_name


def get_folder_path(base_path, r, kappa, use_second_order, wP, algorithm='qlearning'):
    """Helper to construct folder path"""
    folder_name = get_folder_name(r, kappa, use_second_order, 0.8, wP, 1.0, algorithm=algorithm)
    return os.path.join(base_path, folder_name, "data", "experiment_data.h5")


def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--data-dir', type=str, default='.',
                       help='Base directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='paper_figures',
                       help='Output directory for figures')
    parser.add_argument('--figures', type=str, nargs='+',
                       choices=['2', '4', '6', '7', '8', '9', 'all'],
                       default=['all'],
                       help='Which figures to generate')
    parser.add_argument('--algorithm', type=str, default='qlearning',
                       choices=['qlearning', 'sarsa', 'expected_sarsa', 'double_qlearning'],
                       help='Algorithm name to use when constructing data paths')
    
    args = parser.parse_args()
    
    # Setup plotting style
    setup_matplotlib_for_publication(font_size_pt=12)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_path = args.data_dir
    output_dir = args.output_dir
    algorithm = args.algorithm
    
    figures_to_plot = args.figures
    if 'all' in figures_to_plot:
        figures_to_plot = ['2', '4', '6', '7', '8', '9']
    
    # Figure 2
    if '2' in figures_to_plot:
        print(f"\n--- Generating Figure 2 (algorithm: {algorithm}) ---")
        fig2_paths = {
            kappa: {
                'M1': get_folder_path(base_path, 3.6, kappa, False, 1.0, algorithm),
                'M2': get_folder_path(base_path, 3.6, kappa, True, 1.0, algorithm)
            } for kappa in [0.0, 0.5, 1.0, 1.5, 2.0]
        }
        plot_figure_2(fig2_paths, os.path.join(output_dir, "Figure_2.pdf"), 
                     total_iterations=1000000)
    
    # Figure 4
    if '4' in figures_to_plot:
        print(f"\n--- Generating Figure 4 (algorithm: {algorithm}) ---")
        fig4_paths = {
            kappa: get_folder_path(base_path, 3.6, kappa, False, 1.0, algorithm)
            for kappa in [0.0, 0.5, 1.0, 1.5, 2.0]
        }
        plot_figure_4(fig4_paths, os.path.join(output_dir, "Figure_4.pdf"))
    
    # Figure 6
    if '6' in figures_to_plot:
        print(f"\n--- Generating Figure 6 (algorithm: {algorithm}) ---")
        fig6_paths = {
            'M1': {
                'Hybrid': get_folder_path(base_path, 3.0, 1.0, False, 0.95, algorithm),
                'Sole reputation': get_folder_path(base_path, 3.0, 0.0, False, 0.95, algorithm),
                'Sole NI': get_folder_path(base_path, 3.0, 1.0, False, 1.0, algorithm),
            },
            'M2': {
                'Hybrid': get_folder_path(base_path, 3.0, 1.0, True, 0.95, algorithm),
                'Sole reputation': get_folder_path(base_path, 3.0, 0.0, True, 0.95, algorithm),
                'Sole NI': get_folder_path(base_path, 3.0, 1.0, True, 1.0, algorithm),
            }
        }
        plot_figure_6(fig6_paths, os.path.join(output_dir, "Figure_6.pdf"))
    
    # Figure 7
    if '7' in figures_to_plot:
        print(f"\n--- Generating Figure 7 (algorithm: {algorithm}) ---")
        fig7_paths = {
            'Sole reputation': get_folder_path(base_path, 3.0, 0.0, False, 0.95, algorithm),
            'Sole NI': get_folder_path(base_path, 3.0, 1.0, False, 1.0, algorithm),
            'Hybrid mechanism': get_folder_path(base_path, 3.0, 1.0, False, 0.95, algorithm)
        }
        plot_figure_7(fig7_paths, os.path.join(output_dir, "Figure_7.pdf"))
    
    # Figure 8
    if '8' in figures_to_plot:
        print(f"\n--- Generating Figure 8 (algorithm: {algorithm}) ---")
        fig8_paths = {
            'with_ni': {
                1: get_folder_path(base_path, 3.0, 1.0, False, 0.95, algorithm),
                2: get_folder_path(base_path, 3.0, 1.0, True, 0.95, algorithm)
            },
            'no_ni': {
                1: get_folder_path(base_path, 3.0, 0.0, False, 0.95, algorithm),
                2: get_folder_path(base_path, 3.0, 0.0, True, 0.95, algorithm)
            }
        }
        plot_figure_8(fig8_paths, os.path.join(output_dir, "Figure_8.pdf"))
    
    # Figure 9
    if '9' in figures_to_plot:
        print(f"\n--- Generating Figure 9 (algorithm: {algorithm}) ---")
        fig9_path = get_folder_path(base_path, 3.0, 1.0, True, 0.95, algorithm)
        plot_figure_9(fig9_path, os.path.join(output_dir, "Figure_9.pdf"))
    
    print("\nAll plotting tasks are complete.")


if __name__ == '__main__':
    main()

