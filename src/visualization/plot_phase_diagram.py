#!/usr/bin/env python3
"""
Plot Phase Diagram from aggregated experiment results.

Generates a 2D heatmap showing cooperation rate as a function of (lambda, r).
"""
import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.visualization.plot_utils import setup_matplotlib_for_publication


def load_all_results(input_dir):
    """Load all chunk results from the input directory."""
    all_results = []
    
    # Find all JSON result files
    pattern = os.path.join(input_dir, '**/chunk_*.json')
    json_files = glob(pattern, recursive=True)
    
    if not json_files:
        # Try alternative pattern
        pattern = os.path.join(input_dir, '**/chunk_*.json')
        json_files = glob(pattern, recursive=True)
    
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_results


def create_phase_diagram(results, lambda_points, r_points):
    """Create 2D array from results."""
    # Define grid
    lambda_values = np.linspace(0.0, 1.0, lambda_points)
    r_values = np.linspace(2.0, 5.0, r_points)
    
    # Initialize grid with NaN
    grid = np.full((r_points, lambda_points), np.nan)
    
    # Fill grid with results
    for result in results:
        lam = result['lambda']
        r = result['r']
        coop = result['final_coop']
        
        if coop < 0:  # Error marker
            continue
        
        # Find closest indices
        lam_idx = np.argmin(np.abs(lambda_values - lam))
        r_idx = np.argmin(np.abs(r_values - r))
        
        grid[r_idx, lam_idx] = coop
    
    return grid, lambda_values, r_values


def plot_phase_diagram(grid, lambda_values, r_values, output_path, title='Phase Diagram'):
    """Plot the phase diagram as a heatmap."""
    setup_matplotlib_for_publication()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[lambda_values[0], lambda_values[-1], r_values[0], r_values[-1]],
                   cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Final Cooperation Rate')
    
    # Add contour lines
    try:
        contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        cs = ax.contour(lambda_values, r_values, grid, levels=contour_levels,
                       colors='black', linewidths=0.5, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    except:
        pass  # Skip contours if data is incomplete
    
    # Labels
    ax.set_xlabel('λ (DQN Weight)', fontsize=14)
    ax.set_ylabel('r (Synergy Factor)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_phase_curves(grid, lambda_values, r_values, output_path):
    """Plot 1D phase transition curves for selected lambda values."""
    setup_matplotlib_for_publication()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select representative lambda values
    lambda_indices = [0, len(lambda_values)//4, len(lambda_values)//2, 
                      3*len(lambda_values)//4, len(lambda_values)-1]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_indices)))
    
    for idx, lam_idx in enumerate(lambda_indices):
        lam = lambda_values[lam_idx]
        coop_vs_r = grid[:, lam_idx]
        
        # Only plot if we have valid data
        valid = ~np.isnan(coop_vs_r)
        if np.any(valid):
            ax.plot(r_values[valid], coop_vs_r[valid], 
                   'o-', color=colors[idx], label=f'λ={lam:.2f}',
                   linewidth=2, markersize=4)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax.set_xlabel('r (Synergy Factor)', fontsize=14)
    ax.set_ylabel('Final Cooperation Rate', fontsize=14)
    ax.set_title('Phase Transition Curves', fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_lambda_curves(grid, lambda_values, r_values, output_path):
    """Plot 1D curves: cooperation vs lambda for selected r values."""
    setup_matplotlib_for_publication()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select representative r values
    r_indices = [0, len(r_values)//4, len(r_values)//2, 
                 3*len(r_values)//4, len(r_values)-1]
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(r_indices)))
    
    for idx, r_idx in enumerate(r_indices):
        r = r_values[r_idx]
        coop_vs_lambda = grid[r_idx, :]
        
        valid = ~np.isnan(coop_vs_lambda)
        if np.any(valid):
            ax.plot(lambda_values[valid], coop_vs_lambda[valid], 
                   'o-', color=colors[idx], label=f'r={r:.2f}',
                   linewidth=2, markersize=4)
    
    ax.set_xlabel('λ (DQN Weight)', fontsize=14)
    ax.set_ylabel('Final Cooperation Rate', fontsize=14)
    ax.set_title('Effect of λ on Cooperation', fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot phase diagram')
    parser.add_argument('--input', type=str, required=True, help='Input directory with results')
    parser.add_argument('--output', type=str, default='paper_figures/', help='Output directory')
    parser.add_argument('--lambda_points', type=int, default=50, help='Number of lambda points')
    parser.add_argument('--r_points', type=int, default=50, help='Number of r points')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results = load_all_results(args.input)
    print(f"Loaded {len(results)} data points")
    
    if not results:
        print("ERROR: No results found!")
        return
    
    # Create grid
    grid, lambda_values, r_values = create_phase_diagram(results, args.lambda_points, args.r_points)
    
    # Count valid points
    valid_count = np.sum(~np.isnan(grid))
    total_count = grid.size
    print(f"Valid data points: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    
    # Generate plots
    plot_phase_diagram(grid, lambda_values, r_values,
                      os.path.join(args.output, 'phase_diagram_heatmap.png'),
                      title='Cooperation Phase Diagram')
    
    plot_phase_curves(grid, lambda_values, r_values,
                     os.path.join(args.output, 'phase_transition_curves.png'))
    
    plot_lambda_curves(grid, lambda_values, r_values,
                      os.path.join(args.output, 'lambda_effect_curves.png'))
    
    # Save raw data
    np.savez(os.path.join(args.output, 'phase_diagram_data.npz'),
             grid=grid, lambda_values=lambda_values, r_values=r_values)
    
    print("\nPhase diagram generation complete!")


if __name__ == "__main__":
    main()
