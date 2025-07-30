# Filename: plot_combined_comparisons.py
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Global Plotting and Font Settings
# =============================================================================
def setup_matplotlib_for_publication(font_size_pt=12):
    plt.rcParams.update({
        "font.size": font_size_pt, "axes.titlesize": font_size_pt, "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 2, "ytick.labelsize": font_size_pt - 2,
        "legend.fontsize": font_size_pt - 2, "figure.titlesize": font_size_pt + 2,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.usetex": False, "figure.dpi": 300,
    })
    print(f"Matplotlib style updated for {font_size_pt}pt font.")

# =============================================================================
# 2. Data Loading Function
# =============================================================================
def load_data(filepath, dataset_name):
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found at {filepath}")
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name in f: return f[dataset_name][:]
            else:
                print(f"Warning: Dataset '{dataset_name}' not found in {filepath}")
                return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# =============================================================================
# 3. (New) Combined Plotting Function
# =============================================================================
def plot_combined_comparison(data_paths, output_filename, total_iterations=100001):
    """
    Plot two subplots in a single figure: cooperation rate comparison on the left, and strategy switch count comparison on the right.
    """
    # --- Key change: create a figure with 1 row and 2 columns of subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # Adjust figsize to accommodate two plots

    # --- (a) Plot the left subplot: cooperation rate comparison ---
    for label, path in data_paths.items():
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            plot_y = np.full(total_iterations, np.nan)
            plot_y[:len(coop_hist)] = coop_hist
            iterations = np.arange(1, total_iterations + 1)
            ax1.plot(iterations, plot_y, label=label, lw=1.5)

   
    ax1.set_xlabel('$t$ ')
    ax1.set_ylabel('$f_c$')
    ax1.set_xscale('log')
    ax1.set_title('(a)')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(1, total_iterations)
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # --- (b) Plot the right subplot: strategy switch comparison ---
    for label, path in data_paths.items():
        switch_c_d = load_data(path, 'switch_C_to_D')
        switch_d_c = load_data(path, 'switch_D_to_C')
        if switch_c_d is not None and switch_d_c is not None:
            total_switches = switch_c_d + switch_d_c
            iterations = np.arange(1, len(total_switches) + 1)
            ax2.plot(iterations, total_switches, label=label, lw=1.5)
   
    
    ax2.set_xlabel('$t$ ')
    ax2.set_ylabel('Number of Strategy Switches')
    ax2.set_xscale('log')
    ax2.set_title('(b)')
    ax2.set_ylim(-5,10005) # The lower limit of the log axis cannot be 0
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    # --- Uniformly adjust the layout and save ---
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined comparison figure saved to {output_filename}")


# =============================================================================
# 4. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # --- 1. Set plotting style ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. Specify base data and output directories ---
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 3. Define the data paths for the two models to be compared ---
    comparison_paths = {
        'State: Previous Action': os.path.join(BASE_DATA_PATH, "results_r4.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00_action/data/experiment_data.h5"),
        'State: Reputation': os.path.join(BASE_DATA_PATH, "results_r4.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00/data/experiment_data.h5"),
    }
    
    # Total number of simulation iterations, for unifying the x-axis
    TOTAL_ITERATIONS = 100001
    
    # --- 4. Call the unique, combined plotting function ---
    print("\n--- Generating Combined Comparison Figure ---")
    plot_combined_comparison(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Coop_vs_Stability_Comparison.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    
    print("\nAll plotting tasks are complete.")