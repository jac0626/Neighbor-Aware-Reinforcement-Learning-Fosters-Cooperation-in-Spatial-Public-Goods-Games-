# Filename: plot_q_value_figures.py
# (Retain setup_matplotlib_for_publication and load_data functions)
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Global Plotting and Font Settings (consistent with your specifications)
# =============================================================================
def setup_matplotlib_for_publication(font_size_pt=12):
    """
    Configure Matplotlib for publication-quality charts.
    """
    plt.rcParams.update({
        "font.size": font_size_pt, "axes.titlesize": font_size_pt, "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 2, "ytick.labelsize": font_size_pt - 2,
        "legend.fontsize": font_size_pt - 2, "figure.titlesize": font_size_pt + 2,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.usetex": False, # Set to True if LaTeX formulas are needed
        "figure.dpi": 300,
    })
    print(f"Matplotlib style updated for {font_size_pt}pt font.")

# =============================================================================
# 2. Data Loading Function (consistent with your specifications)
# =============================================================================
def load_data(filepath, dataset_name):
    """
    Safely load the specified dataset from an HDF5 file.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found at {filepath}")
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name in f:
                return f[dataset_name][:]
            else:
                print(f"Warning: Dataset '{dataset_name}' not found in {filepath}")
                return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# =============================================================================
# 3. New Q-value evolution plotting function, partitioned by strategy group
# =============================================================================
def plot_q_by_strategy_comparison(data_paths, output_filename, total_iterations=20000):
    """
    Plot multiple panels in a single figure, with each panel containing two subplots (left and right),
    showing the Q-value evolution for cooperator and defector groups, respectively.
    """
    num_param_sets = len(data_paths)
    if num_param_sets == 0: return

    # --- Each parameter setting gets a row with two subplots (cooperators/defectors) ---
    fig, axes = plt.subplots(num_param_sets, 2, 
                             figsize=(10, 4.5 * num_param_sets), 
                             sharey=True) # All subplots share the Y-axis
    
    if num_param_sets == 1: # Ensure axes is a 2D array
        axes = np.array([axes])

    # --- Define styles and labels ---
    plot_styles = {
        'q_s0_c': {'color': 'blue',    'linestyle': '-',  'label': '$Q(s_0, C)$'},
        'q_s0_d': {'color': 'red',     'linestyle': '-',  'label': '$Q(s_0, D)$'},
        'q_s1_c': {'color': '#66c2a5', 'linestyle': '--', 'label': '$Q(s_1, C)$'},
        'q_s1_d': {'color': '#fc8d62', 'linestyle': '--', 'label': '$Q(s_1, D)$'}
    }
    q_keys_map = {key: f"_{key}_history" for key in plot_styles}

    # --- Iterate over each parameter setting (corresponding to each row) ---
    for i, (row_label, path) in enumerate(data_paths.items()):
        ax_coop = axes[i, 0]
        ax_def = axes[i, 1]

        # --- (a) Plot the left subplot: Q-values of the cooperator group ---
        for key, hdf5_suffix in q_keys_map.items():
            dataset_name = f"cooperators{hdf5_suffix}"
            data = load_data(path, dataset_name)
            if data is not None:
                iterations = np.arange(1, len(data) + 1)
                ax_coop.plot(iterations, data, lw=1.5, **plot_styles[key])
        
        # --- (b) Plot the right subplot: Q-values of the defector group ---
        for key, hdf5_suffix in q_keys_map.items():
            dataset_name = f"defectors{hdf5_suffix}"
            data = load_data(path, dataset_name)
            if data is not None:
                iterations = np.arange(1, len(data) + 1)
                ax_def.plot(iterations, data, lw=1.5, **plot_styles[key])

        # --- Format subplots ---
        for ax in [ax_coop, ax_def]:
            ax.set_xscale('log')
            ax.set_xlim(1, total_iterations)
            ax.grid(True, which="both", ls="--", alpha=0.6)
            if i == num_param_sets - 1: # Only label the x-axis on the bottom-most subplots
                ax.set_xlabel('$t$ (Iteration)')
        
        ax_coop.set_ylabel('Average Q-value')
        
        # Set the row title (at the ylabel position of the left subplot)
        ax_coop.set_ylabel(f"{row_label}\n\nAverage Q-value", multialignment='center')


    # Set column titles
    axes[0, 0].set_title('Q-values of Cooperators')
    axes[0, 1].set_title('Q-values of Defectors')

    # --- Global legend and layout ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.94]) # Adjust layout to make space for the legend
    
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Q-value by strategy figure saved to {output_filename}")


# =============================================================================
# 4. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    setup_matplotlib_for_publication(font_size_pt=12)
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    FIXED_PARAMS = {
        'wP': 0.83, 'kappa': 0.0, 'M_bool': False,
        'alpha': 0.8, 'rep_gain_C': 1.0
    }
    r_values_to_compare = [1.0, 2.0,3.0] 

    def get_folder_name(r, kappa, M_bool, wP, alpha, rep_gain_C):
        order_str = str(M_bool)
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha{alpha}_rw{wP:.2f}_rgC{rep_gain_C:.2f}"

    comparison_paths = {
        f"$r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, **FIXED_PARAMS), 
            "data/experiment_data.h5"
        )
        for r_val in r_values_to_compare
    }

    TOTAL_ITERATIONS = 20001

    print("\n--- Generating Q-value by Strategy Comparison Figure ---")
    plot_q_by_strategy_comparison(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Q_Value_by_Strategy.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    print("\nPlotting task is complete.")