import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plot_utils import setup_matplotlib_for_publication

def load_data(results_dir):
    """
    Scans the results directory for HDF5 files and aggregates data.
    Returns a structured dictionary: data[r][lambda] = { 'coop_rate': [], 'final_coop': float }
    """
    data = {}
    # Pattern: results-r{r}-inf{inf}-lam{lam}/data/experiment_data.h5
    # Note: GitHub artifacts might be unzipped into folders
    search_path = os.path.join(results_dir, "**", "experiment_data.h5")
    files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} data files.")
    
    for filepath in files:
        try:
            # Extract parameters from folder name
            # Folder format expected: .../r{r}_inf{inf}_lam{lam}_{state_type}/...
            # But the artifact download might change structure.
            # Let's rely on reading config from HDF5 attributes if possible, or parse path carefully.
            # Since we don't save config in HDF5 attributes yet (we should!), we parse path.
            # Path example: results/r3.0_inf1.0_lam0.1_reputation/data/experiment_data.h5
            
            parts = filepath.split(os.sep)
            # Find the folder that starts with 'r' and contains 'lam'
            # We iterate through all parts to find one that matches the pattern
            param_folder = None
            for p in parts:
                if p.startswith('r') and 'lam' in p and 'inf' in p:
                     param_folder = p
                     break
            
            if not param_folder:
                print(f"Skipping {filepath}: Cannot find parameter folder in path {filepath}")
                continue
                
            # Parse r and lambda
            # Format: r3.0_inf1.0_lam0.1_reputation
            # We look for the segment that matches this pattern
            segments = param_folder.split('_')
            
            # Helper to safely extract float value after a prefix
            def get_val(segs, prefix):
                for s in segs:
                    if s.startswith(prefix):
                        try:
                            return float(s[len(prefix):])
                        except ValueError:
                            continue
                return None

            r_val = get_val(segments, 'r')
            lam_val = get_val(segments, 'lam')
            
            if r_val is None or lam_val is None:
                 print(f"Skipping {filepath}: Could not parse r or lam from {param_folder}")
                 continue
            
            if r_val not in data:
                data[r_val] = {}
            if lam_val not in data[r_val]:
                data[r_val][lam_val] = []
                
            with h5py.File(filepath, 'r') as f:
                coop_rate = f['coop_rate_history'][:]
                data[r_val][lam_val].append(coop_rate)
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return data

def plot_cooperation_evolution(data, target_r, output_dir):
    """
    Plot 1: Cooperation Rate Evolution (Single vs Dual) at fixed r.
    """
    if target_r not in data:
        print(f"Target r={target_r} not found in data.")
        return

    setup_matplotlib_for_publication()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Baseline (Lambda=0)
    if 0.0 in data[target_r]:
        runs = data[target_r][0.0]
        # Average over runs if multiple (though matrix usually runs 1 per param set unless repeated)
        # Assuming 1 run per param set for now, or we average if multiple
        avg_run = np.mean(runs, axis=0)
        ax.plot(avg_run, label='Single-Brain (Baseline)', color='gray', linestyle='--', linewidth=2)
        
    # Dual-Brain (Lambda=0.1, 0.3, 0.5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, lam in enumerate(sorted([k for k in data[target_r].keys() if k > 0])):
        runs = data[target_r][lam]
        avg_run = np.mean(runs, axis=0)
        ax.plot(avg_run, label=f'Dual-Brain ($\lambda={lam}$)', color=colors[i % len(colors)], linewidth=2)
        
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(f'Cooperation Evolution (r={target_r})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_evolution_comparison.png'), dpi=300)
    plt.close()
    print("Generated Plot 1: Evolution Comparison")

def plot_robustness_analysis(data, output_dir):
    """
    Plot 2: Robustness (Coop Rate vs r) for Single vs Dual.
    """
    setup_matplotlib_for_publication()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    r_values = sorted(data.keys())
    
    # Extract final coop rates
    y_single = []
    y_dual_best = [] # Best dual brain performance (e.g. lambda=0.1)
    
    for r in r_values:
        # Single Brain
        if 0.0 in data[r]:
            final_coop = np.mean([run[-1] for run in data[r][0.0]])
            y_single.append(final_coop)
        else:
            y_single.append(np.nan)
            
        # Dual Brain (pick lambda=0.1 as representative)
        if 0.1 in data[r]:
            final_coop = np.mean([run[-1] for run in data[r][0.1]])
            y_dual_best.append(final_coop)
        else:
            y_dual_best.append(np.nan)
            
    ax.plot(r_values, y_single, 'o--', label='Single-Brain', color='gray', markersize=8)
    ax.plot(r_values, y_dual_best, 's-', label='Dual-Brain ($\lambda=0.1$)', color='#d62728', markersize=8)
    
    ax.set_xlabel('Synergy Factor (r)')
    ax.set_ylabel('Final Cooperation Rate')
    ax.set_title('Robustness Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_robustness_analysis.png'), dpi=300)
    plt.close()
    print("Generated Plot 2: Robustness Analysis")

def plot_lambda_impact(data, target_r, output_dir):
    """
    Plot 3: Impact of Lambda on Cooperation Rate at fixed r.
    """
    if target_r not in data:
        print(f"Target r={target_r} not found in data.")
        return
        
    setup_matplotlib_for_publication()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    lambdas = sorted(data[target_r].keys())
    final_coops = []
    
    for lam in lambdas:
        final_coop = np.mean([run[-1] for run in data[target_r][lam]])
        final_coops.append(final_coop)
        
    ax.bar([str(l) for l in lambdas], final_coops, color='#17becf', alpha=0.7)
    ax.plot([str(l) for l in lambdas], final_coops, 'r-o', linewidth=2)
    
    ax.set_xlabel('DQN Mixing Coefficient ($\lambda$)')
    ax.set_ylabel('Final Cooperation Rate')
    ax.set_title(f'Impact of Dual-Brain Weight (r={target_r})')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_lambda_impact.png'), dpi=300)
    plt.close()
    print("Generated Plot 3: Lambda Impact")

def plot_evolution_grid(data, output_dir):
    """
    Plot 4: Grid of cooperation evolution at multiple r values.
    Shows low, medium, high dilemma scenarios.
    """
    available_rs = sorted(data.keys())
    if len(available_rs) < 3:
        print("Not enough r values for grid plot.")
        return
    
    # Select representative r values: low (easy), medium, high (hard)
    # Low r: cooperation harder (less synergy)
    # High r: cooperation easier (more synergy)
    low_r = available_rs[0]
    mid_r = available_rs[len(available_rs)//2]
    high_r = available_rs[-1]
    
    # Also pick one more intermediate value if available
    if len(available_rs) >= 5:
        mid_low_r = available_rs[len(available_rs)//4]
        selected_rs = [low_r, mid_low_r, mid_r, high_r]
    else:
        selected_rs = [low_r, mid_r, high_r]
    
    setup_matplotlib_for_publication()
    n_plots = len(selected_rs)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors_dual = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, r_val in enumerate(selected_rs):
        ax = axes[idx]
        
        # Baseline (Lambda=0)
        if 0.0 in data[r_val]:
            runs = data[r_val][0.0]
            avg_run = np.mean(runs, axis=0)
            ax.plot(avg_run, label='Single-Brain (λ=0)', color='gray', linestyle='--', linewidth=2.5, alpha=0.8)
        
        # Dual-Brain
        dual_lambdas = sorted([k for k in data[r_val].keys() if k > 0])
        for i, lam in enumerate(dual_lambdas):
            runs = data[r_val][lam]
            avg_run = np.mean(runs, axis=0)
            ax.plot(avg_run, label=f'λ={lam}', color=colors_dual[i % len(colors_dual)], linewidth=2)
        
        ax.set_xlabel('Iterations', fontsize=11)
        ax.set_ylabel('Cooperation Rate', fontsize=11)
        ax.set_title(f'r = {r_val}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Add text annotation for difficulty
        if r_val == low_r:
            ax.text(0.05, 0.95, 'Low Synergy\n(Hard)', transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif r_val == high_r:
            ax.text(0.05, 0.95, 'High Synergy\n(Easy)', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Hide extra subplot if we have only 3
    if n_plots < 4:
        axes[3].axis('off')
    
    plt.suptitle('Cooperation Evolution Across Different Synergy Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_evolution_grid.png'), dpi=300)
    plt.close()
    print("Generated Plot 4: Evolution Grid")

def plot_heatmap_comparison(data, output_dir):
    """
    Plot 5: Heatmap showing final cooperation rate for all (r, lambda) combinations.
    """
    setup_matplotlib_for_publication()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    r_values = sorted(data.keys())
    all_lambdas = set()
    for r in r_values:
        all_lambdas.update(data[r].keys())
    lambda_values = sorted(all_lambdas)
    
    # Create matrix for final cooperation rates
    coop_matrix = np.zeros((len(lambda_values), len(r_values)))
    
    for i, lam in enumerate(lambda_values):
        for j, r in enumerate(r_values):
            if lam in data[r]:
                final_coop = np.mean([run[-1] for run in data[r][lam]])
                coop_matrix[i, j] = final_coop
            else:
                coop_matrix[i, j] = np.nan
    
    # Plot 1: Heatmap
    im1 = ax1.imshow(coop_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xticks(range(len(r_values)))
    ax1.set_xticklabels([f'{r:.1f}' for r in r_values], rotation=45)
    ax1.set_yticks(range(len(lambda_values)))
    ax1.set_yticklabels([f'{lam:.1f}' for lam in lambda_values])
    ax1.set_xlabel('Synergy Factor (r)', fontsize=12)
    ax1.set_ylabel('DQN Weight (λ)', fontsize=12)
    ax1.set_title('Final Cooperation Rate Heatmap', fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(len(lambda_values)):
        for j in range(len(r_values)):
            if not np.isnan(coop_matrix[i, j]):
                text = ax1.text(j, i, f'{coop_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Cooperation Rate')
    
    # Plot 2: Line plot showing trends
    for i, lam in enumerate(lambda_values):
        y_vals = coop_matrix[i, :]
        ax2.plot(r_values, y_vals, 'o-', label=f'λ={lam:.1f}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Synergy Factor (r)', fontsize=12)
    ax2.set_ylabel('Final Cooperation Rate', fontsize=12)
    ax2.set_title('Cooperation vs Synergy by λ', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_heatmap_comparison.png'), dpi=300)
    plt.close()
    print("Generated Plot 5: Heatmap Comparison")

def main():
    results_dir = "all_results" # Folder where artifacts are downloaded
    output_dir = "proposal_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    data = load_data(results_dir)
    
    if not data:
        print("No data found. Exiting.")
        return
    
    available_rs = sorted(data.keys())
    print(f"Available r values: {available_rs}")
    
    # Select target r for single plots
    target_r = 4.0  # Changed from 3.0 to 4.0
    if target_r not in data and available_rs:
        target_r = available_rs[len(available_rs)//2]
        print(f"r=4.0 not found, using r={target_r}")
    
    # Generate Original 3 Plots (focused on r=4.0 or middle value)
    print("\nGenerating focused plots...")
    plot_cooperation_evolution(data, target_r=target_r, output_dir=output_dir)
    plot_robustness_analysis(data, output_dir=output_dir)
    plot_lambda_impact(data, target_r=target_r, output_dir=output_dir)
    
    # Generate NEW plots (covering multiple r values)
    print("\nGenerating comprehensive plots...")
    plot_evolution_grid(data, output_dir=output_dir)
    plot_heatmap_comparison(data, output_dir=output_dir)
    
    print(f"\n✅ All 5 plots saved to {output_dir}/")
    print("  1. Evolution Comparison (single r)")
    print("  2. Robustness Analysis (all r)")
    print("  3. Lambda Impact (single r)")
    print("  4. Evolution Grid (multiple r) 🆕")
    print("  5. Heatmap Comparison (all r×λ) 🆕")

if __name__ == "__main__":
    main()
