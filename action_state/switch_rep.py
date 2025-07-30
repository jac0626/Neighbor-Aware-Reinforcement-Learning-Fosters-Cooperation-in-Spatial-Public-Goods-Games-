# Filename: plot_all_figures_with_switches.py
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
# 3. Standalone Plotting Functions
# =============================================================================

# --- (Paste all previous plot_figure_X functions here: plot_figure_2, 3, 4, 6, 7, 8, 9) ---
def plot_figure_2(data_paths, output_filename, total_iterations=200000):
    # ... (code from the previous version) ...
    pass
def plot_figure_3(data_paths, output_filename):
    # ... (code from the previous version) ...
    pass
# ... (other plotting functions) ...

# --- (New) Strategy Switch Plotting Function ---
# --- (Modified) Strategy Switch Plotting Function ---
def plot_strategy_switches(data_paths, output_filename):
    """
    Plot the evolution of the number of strategy switches over time to show system stability.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for label, path in data_paths.items():
        switch_c_d = load_data(path, 'switch_C_to_D')
        switch_d_c = load_data(path, 'switch_D_to_C')
        
        if switch_c_d is not None and switch_d_c is not None:
            total_switches = switch_c_d + switch_d_c
            
            # --- For better visual effect, we plot the raw number of switches instead of the percentage ---
            # because the percentage can be very small
            iterations = np.arange(1, len(total_switches) + 1)
            ax.plot(iterations, total_switches, label=label, lw=1.5) # lw is line width

    ax.set_title('Comparison of Model Stability')
    ax.set_xlabel('$t$ (Iteration)')
    ax.set_ylabel('Number of Strategy Switches')
    
    # --- Key change: use a log-log scale ---
    # This is very effective for showing the number of switches across multiple orders of magnitude
    ax.set_xscale('log')
    
    # Set a reasonable lower limit for the Y-axis to avoid log issues with zero values
   

    ax.legend(title="State Definition")
    ax.grid(True, which="both", ls="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Strategy switch figure saved to {output_filename}")

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

    # --- 3. (New) Plot strategy switch comparison figure ---
    print("\n--- Generating Strategy Switch Comparison Figure ---")
    
    # --- Key change: change set to dict and provide clear labels ---
    fig_switch_paths = {
        'State: Previous Action': os.path.join(BASE_DATA_PATH, "results_r4.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00_action/data/experiment_data.h5"),
        'State: Reputation': os.path.join(BASE_DATA_PATH, "results_r4.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00/data/experiment_data.h5"),
    }
    
    plot_strategy_switches(fig_switch_paths, os.path.join(OUTPUT_DIR, "Figure_Stability_Comparison.pdf"))
    
    print("\nAll plotting tasks are complete.")