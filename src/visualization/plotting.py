"""
Plotting Functions for SPGG Experiments
Provides functions for generating publication-quality figures
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def setup_matplotlib_for_publication(font_size_pt=12):
    """
    Set a uniform Matplotlib style for publication-quality figures.
    
    Parameters:
    -----------
    font_size_pt : int
        Base font size in points
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


def load_data(filepath, dataset_name):
    """
    Load a specific dataset from the given HDF5 file.
    
    Parameters:
    -----------
    filepath : str
        Path to HDF5 file
    dataset_name : str
        Name of dataset to load
        
    Returns:
    --------
    numpy.ndarray or None : Dataset array or None if not found
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


def plot_figure_2(data_paths, output_filename, total_iterations=1000000):
    """
    Plot Figure 2: Cooperation rate evolution for different kappa values
    
    Parameters:
    -----------
    data_paths : dict
        Dictionary mapping kappa values to dicts with 'M1' and 'M2' paths
    output_filename : str
        Output file path
    total_iterations : int
        Maximum iterations for x-axis limit
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    kappa_values = sorted(data_paths.keys())

    for kappa in kappa_values:
        path1 = data_paths[kappa]['M1']
        coop_hist1 = load_data(path1, 'coop_rate_history')
        if coop_hist1 is not None:
            ax1.plot(np.arange(1, len(coop_hist1) + 1), coop_hist1, label=f"$\\kappa={kappa}$")
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
            ax2.plot(np.arange(1, len(coop_hist2) + 1), coop_hist2, label=f"$\\kappa={kappa}$")
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


def plot_figure_4(data_paths, output_filename):
    """
    Plot Figure 4: Neighbor influence contribution percentage
    
    Parameters:
    -----------
    data_paths : dict
        Dictionary mapping kappa values to data file paths
    output_filename : str
        Output file path
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    kappa_values = sorted(data_paths.keys())

    for kappa in kappa_values:
        path = data_paths[kappa]
        ni_pct = load_data(path, 'neighbor_influence_percent')
        if ni_pct is not None:
            ax.plot(np.arange(1, len(ni_pct) + 1), ni_pct, label=f"$\\kappa={kappa}$")

    ax.set_xlabel('$t$ ')
    ax.set_ylabel(r'NI Contribution (%)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 4 saved to {output_filename}")


def plot_figure_6(data_paths, output_filename):
    """
    Plot Figure 6: Comparison of different mechanisms
    
    Parameters:
    -----------
    data_paths : dict
        Dictionary with 'M1' and 'M2' keys, each containing dicts with mechanism labels
    output_filename : str
        Output file path
    """
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


def plot_figure_7(data_paths, output_filename):
    """
    Plot Figure 7: Reputation snapshots at different time points
    
    Parameters:
    -----------
    data_paths : dict
        Dictionary mapping mechanism labels to data file paths
    output_filename : str
        Output file path
    """
    time_points = [100, 1000, 10000]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7.5), 
                            gridspec_kw={'hspace': 0.4, 'wspace': 0.1})
    cmap = plt.cm.viridis
    vmin, vmax = -10, 10
    
    for i, (label, path) in enumerate(data_paths.items()):
        axes[i, 0].set_ylabel(label, fontsize=plt.rcParams['axes.labelsize'], 
                             rotation=90, labelpad=20)
        for j, t in enumerate(time_points):
            ax = axes[i, j]
            snapshot = load_data(path, f"R_snapshot_{t}")
            if snapshot is not None:
                im = ax.imshow(snapshot, cmap=cmap, vmin=vmin, vmax=vmax, 
                              interpolation='nearest')
            else:
                ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', fontsize=8)
            if i == 0:
                ax.set_title(f"$t={t}$")
            ax.set_xticks([])
            ax.set_yticks([])
            
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Reputation Value')

    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 7 saved to {output_filename}")


def plot_figure_8(data_paths, output_filename):
    """
    Plot Figure 8: Cooperation rate with and without neighbor influence
    
    Parameters:
    -----------
    data_paths : dict
        Dictionary with 'with_ni' and 'no_ni' keys, each containing dicts with M values
    output_filename : str
        Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

    for M, path in sorted(data_paths['with_ni'].items()):
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            ax1.plot(np.arange(1, len(coop_hist) + 1), coop_hist, label=f"$M={M}$")
    ax1.set_title(r'(a) with NI ($\kappa=1.0$)')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$f_c$')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.6)

    for M, path in sorted(data_paths['no_ni'].items()):
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            ax2.plot(np.arange(1, len(coop_hist) + 1), coop_hist, label=f"$M={M}$")
    ax2.set_title(r'(b) no NI ($\kappa=0$)')
    ax2.set_xlabel('$t$')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 8 saved to {output_filename}")


def plot_figure_9(data_filepath, output_filename):
    """
    Plot Figure 9: Percentage of second-order neighbor influence
    
    Parameters:
    -----------
    data_filepath : str
        Path to data file
    output_filename : str
        Output file path
    """
    data_to_plot = load_data(data_filepath, "best_neighbor_second_order_percent")
    if data_to_plot is None:
        return
    
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


def plot_state_comparison(data_paths, output_filename, total_iterations=100001):
    """
    Plot comparison between different state representations (reputation vs action)
    
    Parameters:
    -----------
    data_paths : dict
        Dictionary mapping labels to data file paths
    output_filename : str
        Output file path
    total_iterations : int
        Maximum iterations for x-axis limit
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) Cooperation rate comparison
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
    ax1.set_title('(a) Cooperation Rate')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(1, total_iterations)
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    
    # (b) Strategy switch comparison
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
    ax2.set_title('(b) System Stability')
    ax2.set_ylim(-5, 10005)
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"State comparison figure saved to {output_filename}")

