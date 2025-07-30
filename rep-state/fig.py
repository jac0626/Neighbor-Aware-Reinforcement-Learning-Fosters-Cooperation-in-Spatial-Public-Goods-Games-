import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Global Plotting and Font Settings
# =============================================================================
def setup_matplotlib_for_publication(font_size_pt=12):
    """
    Set a uniform Matplotlib style for publication-quality figures.
    """
    plt.rcParams.update({
        "font.size": font_size_pt,
        "axes.titlesize": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 2,
        "ytick.labelsize": font_size_pt - 2,
        "legend.fontsize": font_size_pt - 2,
        "figure.titlesize": font_size_pt + 2,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.usetex": False, 
        "figure.dpi": 300,
    })
    print(f"Matplotlib style updated for {font_size_pt}pt font.")

# =============================================================================
# 2. Data Loading Function
# =============================================================================
def load_data(filepath, dataset_name):
    """
    Load a specific dataset from the given HDF5 file.
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
# 3. Standalone Plotting Functions
# =============================================================================

# --- Figure 2 Plotting Function ---
def plot_figure_2(data_paths, output_filename, total_iterations=1000000):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    kappa_values = sorted(data_paths.keys())

    for kappa in kappa_values:
        path1 = data_paths[kappa]['M1']
        coop_hist1 = load_data(path1, 'coop_rate_history')
        if coop_hist1 is not None:
            ax1.plot(np.arange(1, len(coop_hist1) + 1), coop_hist1, label=f"$\kappa={kappa}$")
    ax1.set_title('(a) $M=1$')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$f_c$')
    ax1.set_xscale('log')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(1, total_iterations)
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    for kappa in kappa_values:
        path2 = data_paths[kappa]['M2']
        coop_hist2 = load_data(path2, 'coop_rate_history')
        if coop_hist2 is not None:
            ax2.plot(np.arange(1, len(coop_hist2) + 1), coop_hist2, label=f"$\kappa={kappa}$")
    ax2.set_title('(b) $M=2$')
    ax2.set_xlabel('$t$')
    ax2.set_xscale('log')
    ax2.set_xlim(1, total_iterations)
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 2 saved to {output_filename}")





# --- Figure 4 Plotting Function ---
def plot_figure_4(data_paths, output_filename):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    kappa_values = sorted(data_paths.keys())

    for kappa in kappa_values:
        path = data_paths[kappa]
        ni_pct = load_data(path, 'neighbor_influence_percent')
        if ni_pct is not None:
            ax.plot(np.arange(1, len(ni_pct) + 1), ni_pct, label=f"$\kappa={kappa}$")

    ax.set_xlabel('$t$ ')
    ax.set_ylabel(r'NI Contribution (%)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 4 saved to {output_filename}")


# --- Figure 6 Plotting Function ---
def plot_figure_6(data_paths, output_filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    labels_colors = {'Hybrid': 'C0', 'Sole reputation': 'C2', 'Sole NI': 'C1'}
    
    for label, path in data_paths['M1'].items():
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            ax1.plot(np.arange(1, len(coop_hist) + 1), coop_hist, label=label, color=labels_colors[label])
    ax1.set_title('(a) $M=1$')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$f_c$')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    for label, path in data_paths['M2'].items():
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            ax2.plot(np.arange(1, len(coop_hist) + 1), coop_hist, label=label, color=labels_colors[label])
    ax2.set_title('(b) $M=2$')
    ax2.set_xlabel('$t$')
    ax2.set_xscale('log')
    
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 6 saved to {output_filename}")


# --- Figure 7 Plotting Function ---
def plot_figure_7(data_paths, output_filename):
    time_points = [100, 1000, 10000]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7.5), gridspec_kw={'hspace': 0.4, 'wspace': 0.1})
    cmap = plt.cm.viridis
    vmin, vmax = -10, 10
    
    for i, (label, path) in enumerate(data_paths.items()):
        axes[i, 0].set_ylabel(label, fontsize=plt.rcParams['axes.labelsize'], rotation=90, labelpad=20)
        for j, t in enumerate(time_points):
            ax = axes[i, j]
            snapshot = load_data(path, f"R_snapshot_{t}")
            if snapshot is not None:
                im = ax.imshow(snapshot, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            else:
                ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=8)
            if i == 0: ax.set_title(f"$t={t}$")
            ax.set_xticks([]); ax.set_yticks([])
            
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Reputation Value')

    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 7 saved to {output_filename}")


# --- Figure 8 Plotting Function ---
def plot_figure_8(data_paths, output_filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

    for M, path in sorted(data_paths['with_ni'].items()):
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None: ax1.plot(np.arange(1, len(coop_hist) + 1), coop_hist, label=f"$M={M}$")
    ax1.set_title(r'(a) with NI ($\kappa=1.0$)')
    ax1.set_xlabel('$t$'); ax1.set_ylabel('$f_c$'); ax1.set_xscale('log'); ax1.legend(); ax1.grid(True, ls='--', alpha=0.6)

    for M, path in sorted(data_paths['no_ni'].items()):
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None: ax2.plot(np.arange(1, len(coop_hist) + 1), coop_hist, label=f"$M={M}$")
    ax2.set_title(r'(b) no NI ($\kappa=0$)')
    ax2.set_xlabel('$t$'); ax2.set_xscale('log'); ax2.legend(); ax2.grid(True, ls='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 8 saved to {output_filename}")


# --- Figure 9 Plotting Function ---
def plot_figure_9(data_filepath, output_filename):
    data_to_plot = load_data(data_filepath, "best_neighbor_second_order_percent")
    if data_to_plot is None: return
    
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(np.arange(1, len(data_to_plot) + 1), data_to_plot)
    ax.set_xscale('log')
    ax.set_xlabel('$t$ ')
    ax.set_ylabel(r'NI from 2nd-order neighbors (%)')
    ax.set_xlim(left=1)
    ax.set_ylim(50, 80)
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 9 saved to {output_filename}")


# =============================================================================
# 4. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # --- 1. Set global plotting style ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. Specify base data and output directories ---
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 3. Configure data paths for each figure and call plotting functions ---
    
    # Helper function to construct folder names
    def get_folder_name(r, kappa, M_bool, wP):
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{M_bool}_alpha0.8_rw{wP:.2f}_rgC1.00"

    # --- Plot Figure 2 ---
    print("\n--- Generating Figure 2 ---")
    fig2_paths = {
        kappa: {
            'M1': os.path.join(BASE_DATA_PATH, get_folder_name(3.6, kappa, False, 1.0), "data/experiment_data.h5"),
            'M2': os.path.join(BASE_DATA_PATH, get_folder_name(3.6, kappa, True, 1.0), "data/experiment_data.h5")
        } for kappa in [0.0, 0.5, 1.0, 1.5, 2.0]
    }
    plot_figure_2(fig2_paths, os.path.join(OUTPUT_DIR, "Figure_2.pdf"), total_iterations=1000000)



    # --- Plot Figure 4 ---
    print("\n--- Generating Figure 4 ---")
    fig4_paths = {
        kappa: os.path.join(BASE_DATA_PATH, get_folder_name(3.6, kappa, False, 1.0), "data/experiment_data.h5")
        for kappa in [0.0,0.5, 1.0, 1.5, 2.0]
    }
    plot_figure_4(fig4_paths, os.path.join(OUTPUT_DIR, "Figure_4.pdf"))
    
    # --- Plot Figure 6 ---
    print("\n--- Generating Figure 6 ---")
    fig6_paths = {
        'M1': { 'Hybrid': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, False, 0.95), "data/experiment_data.h5"),
                'Sole reputation': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 0.0, False, 0.95), "data/experiment_data.h5"),
                'Sole NI': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, False, 1.0), "data/experiment_data.h5"), },
        'M2': { 'Hybrid': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, True, 0.95), "data/experiment_data.h5"),
                'Sole reputation': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 0.0, True, 0.95), "data/experiment_data.h5"),
                'Sole NI': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, True, 1.0), "data/experiment_data.h5"), }
    }
    plot_figure_6(fig6_paths, os.path.join(OUTPUT_DIR, "Figure_6.pdf"))
    
    # --- Plot Figure 7 ---
    print("\n--- Generating Figure 7 ---")
    fig7_paths = {
        'Sole reputation': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 0.0, False, 0.95), "data/experiment_data.h5"),
        'Sole NI': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, False, 1.0), "data/experiment_data.h5"),
        'Hybrid mechanism': os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, False, 0.95), "data/experiment_data.h5")
    }
    plot_figure_7(fig7_paths, os.path.join(OUTPUT_DIR, "Figure_7.pdf"))

    # --- Plot Figure 8 ---
    print("\n--- Generating Figure 8 ---")
    fig8_paths = {
        'with_ni': {
            1: os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, False, 0.95), "data/experiment_data.h5"),
            2: os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, True, 0.95), "data/experiment_data.h5") },
        'no_ni': {
            1: os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 0.0, False, 0.95), "data/experiment_data.h5"),
            2: os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 0.0, True, 0.95), "data/experiment_data.h5") }
    }
    plot_figure_8(fig8_paths, os.path.join(OUTPUT_DIR, "Figure_8.pdf"))

    # --- Plot Figure 9 ---
    print("\n--- Generating Figure 9 ---")
    fig9_path = os.path.join(BASE_DATA_PATH, get_folder_name(3.0, 1.0, True, 0.95), "data/experiment_data.h5")
    plot_figure_9(fig9_path, os.path.join(OUTPUT_DIR, "Figure_9.pdf"))
    
    print("\nAll plotting tasks are complete.")