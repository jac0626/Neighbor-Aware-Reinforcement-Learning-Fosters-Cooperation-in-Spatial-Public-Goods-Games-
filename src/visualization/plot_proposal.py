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
            param_folder = next((p for p in parts if p.startswith('r') and 'lam' in p), None)
            
            if not param_folder:
                print(f"Skipping {filepath}: Cannot parse parameters from path.")
                continue
                
            # Parse r and lambda
            # Format: r3.0_inf1.0_lam0.1_reputation
            segments = param_folder.split('_')
            r_val = float(segments[0][1:])
            lam_val = float(segments[2][3:])
            
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

def main():
    results_dir = "all_results" # Folder where artifacts are downloaded
    output_dir = "proposal_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    data = load_data(results_dir)
    
    if not data:
        print("No data found. Exiting.")
        return
        
    # Generate Plots
    # 1. Evolution at r=3.0 (Typical dilemma)
    plot_cooperation_evolution(data, target_r=3.0, output_dir=output_dir)
    
    # 2. Robustness across all r
    plot_robustness_analysis(data, output_dir=output_dir)
    
    # 3. Lambda impact at r=3.0
    plot_lambda_impact(data, target_r=3.0, output_dir=output_dir)
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    main()
