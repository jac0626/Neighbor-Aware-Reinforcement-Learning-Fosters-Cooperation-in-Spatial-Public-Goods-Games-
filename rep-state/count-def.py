# Filename: plot_group_distribution_figures.py (New script)
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ... (setup_matplotlib_for_publication and load_data function code remains the same) ...
def setup_matplotlib_for_publication(font_size_pt=12):
    """
    Configure Matplotlib to generate publication-quality charts.
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
# 3. New plotting function: plotting the evolution of group composition distribution
# =============================================================================
def plot_group_distribution_evolution(data_paths, output_filename, total_iterations=20000):
    """
    Create a subplot for each r value in a single figure,
    with each subplot showing the evolution of the proportion of groups with different numbers of defectors.
    """
    num_plots = len(data_paths)
    if num_plots == 0: return

    # --- Create a figure with N rows and 1 column of subplots ---
    fig, axes = plt.subplots(num_plots, 1, 
                             figsize=(8, 5 * num_plots), 
                             sharex=True, sharey=True) # Share X and Y axes
    
    if num_plots == 1:
        axes = [axes]

    # --- Define styles for the 6 curves ---
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    # --- Iterate over each r value (each subplot) ---
    for i, (label, path) in enumerate(data_paths.items()):
        ax = axes[i]
        
        # Loop to plot the 6 curves (from 0 to 5 defectors)
        for num_d in range(6):
            dataset_name = f"group_comp_d{num_d}_history"
            data = load_data(path, dataset_name)
            
            if data is not None:
                iterations = np.arange(1, len(data) + 1)
                ax.plot(iterations, data, lw=2, color=colors[num_d], 
                        label=f'{num_d} Defectors')
        
        # --- Format the subplot ---
        ax.set_xscale('log')
        ax.set_xlim(1, total_iterations)
        ax.set_ylim(-5, 105) # Y-axis is a percentage
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.set_ylabel('Percentage of Groups (%)')
        ax.set_title(label) # e.g., (a) r=1.0
        ax.legend(title="Group Type", loc='center left', bbox_to_anchor=(1, 0.5))

    axes[-1].set_xlabel('$t$ (Iteration)')

    # --- Save and close ---
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for the legend
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Group distribution evolution figure saved to {output_filename}")


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
    
    TOTAL_ITERATIONS = 20001

    print("\n--- Generating Group Distribution Evolution Figure ---")
    plot_group_distribution_evolution(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Group_Distribution_Evolution.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    print("\nPlotting task is complete.")