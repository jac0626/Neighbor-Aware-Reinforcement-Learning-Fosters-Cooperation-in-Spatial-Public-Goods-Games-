# Filename: plot_reward_comparison.py
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
# 3. Plotting Function: Plotting Key Decision Payoff Comparison
# =============================================================================
def plot_reward_comparison(data_paths, output_filename, total_iterations=20000, smoothing_window=100):
    num_plots = len(data_paths)
    if num_plots == 0: return

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4.5 * num_plots), sharex=True, sharey=True)
    if num_plots == 1: axes = [axes]

    for i, (label, path) in enumerate(data_paths.items()):
        ax = axes[i]
        
        # Load data for key curves
        reward_C_in_1D = load_data(path, 'avg_reward_C_in_1D_history')
        reward_D_in_2D = load_data(path, 'avg_reward_D_in_2D_history')
        reward_D_in_1D = load_data(path, 'avg_reward_D_in_1D_history')

        # Smooth the data
        def smooth(data):
            if data is None or smoothing_window <= 1:
                return data
            # Use 'same' mode to keep the length unchanged, which is easier for plotting
            return np.convolve(data, np.ones(smoothing_window)/smoothing_window, mode='same')

        reward_C_in_1D_s = smooth(reward_C_in_1D)
        reward_D_in_2D_s = smooth(reward_D_in_2D)
        reward_D_in_1D_s = smooth(reward_D_in_1D)

        # Plotting
        if reward_C_in_1D_s is not None:
            iterations = np.arange(1, len(reward_C_in_1D_s) + 1)
            ax.plot(iterations, reward_C_in_1D_s, lw=2, color='#1f77b4', 
                    label='Cooperator in 1D Group\n(Status Quo)')
        
        if reward_D_in_2D_s is not None:
            iterations = np.arange(1, len(reward_D_in_2D_s) + 1)
            ax.plot(iterations, reward_D_in_2D_s, lw=2, color='#ff7f0e',
                    label='Defector in 2D Group\n(Outcome of C→D switch)')
        
        # (Optional) Plot the payoff of defectors in a 1D group for reference
        if reward_D_in_1D_s is not None:
            iterations = np.arange(1, len(reward_D_in_1D_s) + 1)
            ax.plot(iterations, reward_D_in_1D_s, lw=2, color='green', linestyle=':',
                    label='Defector in 1D Group\n(Temptation to Defect)')

        # Formatting
        ax.set_xscale('log')
        ax.set_xlim(1, total_iterations)
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.set_ylabel('Avg. Composite Reward ($R_i$)')
        ax.set_title(label)
        ax.legend(loc='best', fontsize='small')

    axes[-1].set_xlabel('$t$ (Iteration)')
    fig.suptitle('Reward Comparison for Critical State Transitions', fontsize=14, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Critical reward comparison figure saved to {output_filename}")


# =============================================================================
# 4. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    setup_matplotlib_for_publication(font_size_pt=12)
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FIXED_PARAMS = {
        'wP': 0.83, 'kappa': 0.0, 'M_bool': False,
        'alpha': 0.8, 'rep_gain_C': 1.00
    }
    r_values_to_compare = [1.0, 2.0, 3.0] 

    def get_folder_name(r, kappa, M_bool, wP, alpha, rep_gain_C):
        order_str = str(M_bool)
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha{alpha}_rw{wP:.2f}_rgC{rep_gain_C:.2f}"

    comparison_paths = {
        f"({chr(ord('a') + i)}) $r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, **FIXED_PARAMS), 
            "data/experiment_data.h5"
        )
        for i, r_val in enumerate(r_values_to_compare)
    }
    
    TOTAL_ITERATIONS = 20000
    SMOOTHING_WINDOW = 200 # A larger smoothing window can help clarify the trend

    print("\n--- Generating Critical Reward Comparison Figure ---")
    plot_reward_comparison(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Critical_Reward_Comparison.pdf"),
        total_iterations=TOTAL_ITERATIONS,
        smoothing_window=SMOOTHING_WINDOW
    )
    
    print("\nPlotting task is complete.")