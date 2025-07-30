# Filename: plot_q_value_figures.py
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
# 3. New Q-value Evolution Comparison Plotting Function (Core Functionality)
# =============================================================================
def plot_q_evolution_comparison(data_paths, output_filename, total_iterations=20000):
    """
    Plot multiple side-by-side subplots in a single figure, each showing the Q-value evolution for a specific parameter setting.
    """
    num_plots = len(data_paths)
    if num_plots == 0:
        print("No data paths provided for plotting.")
        return

    # --- Create a figure with 1 row and N columns of subplots ---
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4.5), sharey=True)
    if num_plots == 1: # Ensure axes is always an iterable list
        axes = [axes]

    # --- Define styles and labels for Q-value curves ---
    plot_styles = {
        'q_s0_c': {'color': 'blue',    'linestyle': '-',  'label': '$Q(s_0, C)$'},
        'q_s0_d': {'color': 'red',     'linestyle': '-',  'label': '$Q(s_0, D)$'},
        'q_s1_c': {'color': '#66c2a5', 'linestyle': '--', 'label': '$Q(s_1, C)$'},
        'q_s1_d': {'color': '#fc8d62', 'linestyle': '--', 'label': '$Q(s_1, D)$'}  # Use softer colors
    }
    q_keys_map = {
        'q_s0_c': 'avg_q_s0_c_history', 'q_s0_d': 'avg_q_s0_d_history',
        'q_s1_c': 'avg_q_s1_c_history', 'q_s1_d': 'avg_q_s1_d_history'
    }

    # --- Iterate through each parameter setting and plot the subplot ---
    for i, (label, path) in enumerate(data_paths.items()):
        ax = axes[i]
        all_data_loaded = True
        
        # Load and plot the 4 Q-value curves
        for key, hdf5_key in q_keys_map.items():
            data = load_data(path, hdf5_key)
            if data is not None:
                # Unify the X-axis length
                plot_y = np.full(total_iterations, np.nan)
                end_idx = min(len(data), total_iterations)
                plot_y[:end_idx] = data[:end_idx]
                iterations = np.arange(1, total_iterations + 1)
                ax.plot(iterations, plot_y, lw=1.5, **plot_styles[key])
            else:
                all_data_loaded = False
        
        # --- Subplot formatting ---
        if all_data_loaded:
            ax.set_xscale('log')
            ax.set_xlim(1, total_iterations)
            ax.grid(True, which="both", ls="--", alpha=0.6)
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='red')
            
        ax.set_title(label) # Use the dictionary key as the subplot title, e.g., "(a) r=1.0"
        ax.set_xlabel('$t$ (Iteration)')

    # --- Global formatting ---
    axes[0].set_ylabel('Average Q-value')

    # Create a shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Make space for the legend
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Q-value comparison figure saved to {output_filename}")


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
    # Fixed parameters for locating folders
    FIXED_PARAMS = {
        'wP': 0.83,
        'kappa': 0.0,
        'M_bool': False,  # Assuming M=2, use_second_order=True
        'alpha': 0.8,
        'rep_gain_C': 1.0
    }
    
    # Variable parameters
    r_values_to_compare = [1.0, 2.0] 

    # Helper function to construct folder names (ensure this matches your naming convention exactly)
    def get_folder_name(r, kappa, M_bool, wP, alpha, rep_gain_C):
        order_str = str(M_bool) # 'True' or 'False'
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha{alpha}_rw{wP:.2f}_rgC{rep_gain_C:.2f}"

    # Build the data path dictionary
    # The dictionary keys will be used as subplot titles
    comparison_paths = {
        f"(a) $r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, **FIXED_PARAMS), 
            "data/experiment_data.h5"
        )
        for i, r_val in enumerate(r_values_to_compare)
                    
    }
    

    # Total number of simulation iterations, for unifying the x-axis
    TOTAL_ITERATIONS = 10001

    # --- 4. Call the plotting function ---
    print("\n--- Generating Q-value Evolution Comparison Figure ---")
    for label, path in comparison_paths.items():
        print(f"  - Plotting for {label} from {path}")
        
    plot_q_evolution_comparison(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Q_Value_Evolution.pdf") # Save as PDF format
    )
    
    print("\nQ-value plotting task is complete.")