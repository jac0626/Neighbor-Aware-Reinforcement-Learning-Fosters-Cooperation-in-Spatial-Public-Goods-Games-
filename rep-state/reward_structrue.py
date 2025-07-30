# Filename: plot_reward_and_coop_side_by_side.py
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
# 3. (Modified) Combined Plotting Function
# =============================================================================
def plot_reward_and_coop_side_by_side(data_paths, output_filename, total_iterations=100000):
    """
    Plot two side-by-side subplots in a single figure:
    Left plot: Comparison of cooperation rate evolution for different r values.
    Right plot: Comparison of the evolution of the reputation reward ratio for different r values.
    """
    # --- Key change: create a figure with 1 row and 2 columns of subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True) # Share the X-axis

    # Iterate over the data for each r value
    for label, path in data_paths.items():
        # --- (a) Plot the left subplot: cooperation rate comparison ---
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            # Ensure consistent X-axis length
            plot_y = np.full(total_iterations, np.nan)
            plot_y[:total_iterations] = coop_hist[:total_iterations]
            iterations = np.arange(1, total_iterations + 1)
            ax1.plot(iterations, plot_y, label=label, lw=1.5)

        # --- (b) Plot the right subplot: reputation reward ratio comparison ---
        ratio_hist = load_data(path, 'reputation_reward_ratio')
        if ratio_hist is not None:
            iterations = np.arange(1, total_iterations + 1)
            ax2.plot(iterations, ratio_hist[:total_iterations], label=label, lw=1.5)

    # --- Uniformly set the format of the left subplot (cooperation rate) ---
    ax1.set_title('(a) Cooperation Dynamics')
    ax1.set_xlabel('$t$ ')
    ax1.set_ylabel('Fraction of Cooperation ($f_c$)')
    ax1.set_xscale('log')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(1, total_iterations)
    ax1.legend(title="$r$ value")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # --- Uniformly set the format of the right subplot (reputation ratio) ---
    ax2.set_title('(b) Reward Structure')
    ax2.set_xlabel('$t$ ')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'Reputation Reward Ratio (%)')
    # ax2.set_ylim(bottom=0) # The Y-axis range can be dynamically adjusted based on the data
    ax2.legend(title="$r$ value")
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    # --- Uniformly adjust the layout and save ---
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Side-by-side comparison figure saved to {output_filename}")


# =============================================================================
# 4. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # --- 1. Set plotting style ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. Specify base data and output directories ---
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 3. Define the parameter combinations to compare ---
    # This is the area where the reviewer pointed out the anomalous phenomenon
    FIXED_WP = 0.83
    # Select a few r values for comparison to show the trend
    r_values_to_compare = [1.0,2.0] 

    # Helper function to construct folder names
    def get_folder_name(r, kappa, M_bool, wP):
        order_str = 'True' if M_bool else 'False'
        # !! Key !!: Ensure the naming convention here exactly matches your folders
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha0.8_rw{wP:.2f}_rgC1.00"

    # Build the data path dictionary
    comparison_paths = {
        f"$r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, kappa=0.0, M_bool=False, wP=FIXED_WP), 
            "data/experiment_data.h5"
        )
        for r_val in r_values_to_compare
    }
    
    # Total number of simulation iterations, for unifying the x-axis
    TOTAL_ITERATIONS = 20001

    # --- 4. Call the plotting function ---
    print("\n--- Generating Side-by-Side Comparison Figure ---")
    plot_reward_and_coop_side_by_side(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Reward_vs_Coop_SideBySide.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    
    print("\nAll plotting tasks are complete.")