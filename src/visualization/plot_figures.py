
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plot_utils import setup_matplotlib_for_publication

def plot_from_data(folder_name: str):
    """
    Reads HDF5 data from the specified folder and generates standard plots.
    """
    setup_matplotlib_for_publication()
    
    filename = os.path.join(folder_name, "data", "experiment_data.h5")
    if not os.path.exists(filename):
        print(f"Data file not found: {filename}")
        return

    with h5py.File(filename, "r") as data_file:
        coop_hist = data_file["coop_rate_history"][:]
        neighbor_pct = data_file["neighbor_influence_percent"][:]
        pay = data_file["payoff_component_history"][:]
        rep = data_file["rep_component_history"][:]
        it = np.arange(1, len(coop_hist) + 1)
        
        plots_dir = os.path.join(folder_name, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Payoff vs Reputation Components
        plt.figure(figsize=(8,6))
        plt.plot(it, pay, label="w_P · P", linestyle='-')
        plt.plot(it, rep, label="w_R · ΔR", linestyle='--')
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward Component")
        plt.title("Payoff vs Reputation Component over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "payoff_vs_rep.png"))
        plt.close()

        # Cooperation Rate Evolution
        plt.figure(figsize=(8,6))
        plt.plot(it, coop_hist, label="Coop Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Cooperation Rate")
        plt.title("Cooperation Rate Evolution")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "coop_rate_evolution.png"))
        plt.close()

        # Neighbor Influence Percentage
        plt.figure(figsize=(8,6))
        plt.plot(it, neighbor_pct, label="Neighbor Influence %", color='purple')
        plt.xlabel("Iteration")
        plt.ylabel("Percentage (%)")
        plt.title("Neighbor Influence on Q-table Updates")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "neighbor_influence_percent.png"))
        plt.close()

        # Percentage of Second-order Best Neighbors
        if "best_neighbor_second_order_percent" in data_file:
            best_neighbor_pct = data_file["best_neighbor_second_order_percent"][:]
            plt.figure(figsize=(8,6))
            plt.plot(it, best_neighbor_pct, label="Second-order best neighbor %")
            plt.xlabel("Iteration")
            plt.ylabel("Percentage (%)")
            plt.title("Percentage of Second-order Best Neighbors")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "second_order_best_neighbor_percent.png"))
            plt.close()

        # Final Reputation Distribution
        if "rep_hist_final" in data_file and "rep_bins_final" in data_file:
            rep_hist_final = data_file["rep_hist_final"][:]
            rep_bins_final = data_file["rep_bins_final"][:]
            plt.figure(figsize=(8,6))
            plt.bar(rep_bins_final[:-1], rep_hist_final, width=np.diff(rep_bins_final), edgecolor='black')
            plt.xlabel("Reputation")
            plt.ylabel("Frequency")
            plt.title("Final Reputation Distribution")
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, "reputation_distribution.png"))
            plt.close()
            
    print(f"Plots saved to {plots_dir}")
