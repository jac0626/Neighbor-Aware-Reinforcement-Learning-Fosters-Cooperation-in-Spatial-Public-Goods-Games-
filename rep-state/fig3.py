import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# Standalone script for plotting Figure 3.
# =============================================================================

def setup_matplotlib_for_publication(font_size_pt=12):
    """
    Set a uniform Matplotlib style for publication-quality figures.
    """
    # Force a non-interactive backend to ensure compatibility.
    mpl.use('Agg')
    
    plt.rcParams.update({
        "font.size": font_size_pt,
        "axes.titlesize": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 2,
        "ytick.labelsize": font_size_pt - 2,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.usetex": False, 
        "figure.dpi": 300,
    })
    print(f"Matplotlib style updated for {font_size_pt}pt font.")

def load_snapshot_data(filepath, time_point):
    """
    Load a specific strategy snapshot dataset from the given HDF5 file.
    """
    dataset_name = f"Sn_snapshot_{time_point}"
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name in f:
                snapshot = f[dataset_name][:]
                # Debugging info: check the data content
                print(f"Loaded snapshot for t={time_point}: shape={snapshot.shape}, min={np.min(snapshot)}, max={np.max(snapshot)}, mean={np.mean(snapshot)}")
                unique, counts = np.unique(snapshot, return_counts=True)
                print(f"Unique values: {unique}, Counts: {counts}")
                return snapshot
            else:
                print(f"Error: Dataset '{dataset_name}' not found in {filepath}")
                return None
    except Exception as e:
        print(f"Error loading snapshot from {filepath}: {e}")
        return None

def main():
    """
    Main execution function: load data and plot Figure 3.
    """
    # --- 1. Set plotting style ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. Configuration: specify data file paths ---
    dir_k0 = "results_r3.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00"
    dir_k0_5 = "results_r3.6_inf0.5_orderFalse_alpha0.8_rw1.00_rgC1.00"
    
    data_paths = {
        '$\kappa=0$': os.path.join(dir_k0, "data", "experiment_data.h5"),
        '$\kappa=0.5$': os.path.join(dir_k0_5, "data", "experiment_data.h5")
    }
    
    # --- 3. Specify output filename ---
    OUTPUT_FILENAME = "Figure_3_standalone.pdf"

    # --- 4. Start plotting ---
    time_points = [100, 1000, 10000]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 4.8), gridspec_kw={'hspace': 0.3, 'wspace': 0.1})
    
    # Define RGB colors (in 0-1 range)
    color_coop = np.array([192, 192, 192]) / 255.0  # Light gray for cooperators (value 0)
    color_defect = np.array([0, 0, 0]) / 255.0      # Black for defectors (value 1)

    for i, (label, path) in enumerate(data_paths.items()):
        axes[i, 0].set_ylabel(label, fontsize=plt.rcParams['axes.labelsize'], rotation=90, labelpad=20)
        
        for j, t in enumerate(time_points):
            ax = axes[i, j]
            snapshot = load_snapshot_data(path, t)
            
            if snapshot is not None:
                # Initialize with red for easier debugging of unexpected values
                rgb_image = np.full((snapshot.shape[0], snapshot.shape[1], 3), [1, 0, 0])
                rgb_image[snapshot == 0] = color_coop
                rgb_image[snapshot == 1] = color_defect
                
                # Debug: check RGB image pixel values
                print(f"rgb_image[0,0] for {label}, t={t}: {rgb_image[0,0]}")
                print(f"rgb_image[-1,-1] for {label}, t={t}: {rgb_image[-1,-1]}")
                
                ax.imshow(rgb_image, interpolation='nearest', aspect='equal')
            else:
                ax.text(0.5, 0.5, 'Data\nMissing', ha='center', va='center', transform=ax.transAxes, fontsize=8, color='red')
            
            # Set titles for the top row
            if i == 0:
                ax.set_title(f"$t={t}$")
            
            # Hide axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

    # Save the figure
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight')
    plt.savefig("Figure_3_standalone.png", bbox_inches='tight')  # Also save a PNG for easy inspection
    plt.close(fig)
    print(f"\nFigure 3 has been generated and saved as: {OUTPUT_FILENAME} and Figure_3_standalone.png")

if __name__ == '__main__':
    main()