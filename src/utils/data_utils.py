
import h5py
import numpy as np
import os
from typing import Dict, Any, List

class DataManager:
    """
    Handles HDF5 file operations for saving simulation data.
    """
    def __init__(self, filename: str):
        self.filename = filename
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def save_final_data(self, 
                        it_records: List[tuple],
                        epsilon_history: List[float],
                        rep_avg_history: List[float],
                        coop_rate_history: List[float],
                        influence_counts: List[int],
                        switch_C_to_D: List[int],
                        switch_D_to_C: List[int],
                        neighbor_influence_percent: List[float],
                        payoff_comp_history: List[float],
                        rep_comp_history: List[float],
                        best_neighbor_type_history: List[float],
                        q_history_agg: Dict[tuple, Dict[str, List[float]]],
                        Sn_final: np.ndarray,
                        R_final: np.ndarray,
                        cluster_sizes: List[int],
                        reputation_reward_ratio: List[float],
                        avg_reward_C_history: List[float],
                        avg_reward_D_history: List[float],
                        group_composition_history: List[List[float]],
                        q_history_by_strategy: Dict[str, Dict[str, List[float]]],
                        avg_q_history: Dict[str, List[float]],
                        rep_hist_final: np.ndarray,
                        rep_bins_final: np.ndarray,
                        track_positions: List[tuple]
                        ):
        """
        Saves all collected data to the HDF5 file.
        """
        with h5py.File(self.filename, "w") as data_file:
            data_file.create_dataset("it_records_final", data=np.array(it_records))
            data_file.create_dataset("epsilon_history_final", data=np.array(epsilon_history))
            data_file.create_dataset("rep_avg_history_final", data=np.array(rep_avg_history))
            data_file.create_dataset("coop_rate_history", data=np.array(coop_rate_history))
            data_file.create_dataset("influence_counts", data=np.array(influence_counts))
            data_file.create_dataset("switch_C_to_D", data=np.array(switch_C_to_D))
            data_file.create_dataset("switch_D_to_C", data=np.array(switch_D_to_C))
            data_file.create_dataset("neighbor_influence_percent", data=np.array(neighbor_influence_percent))
            data_file.create_dataset("payoff_component_history", data=np.array(payoff_comp_history))
            data_file.create_dataset("rep_component_history", data=np.array(rep_comp_history))
            data_file.create_dataset("best_neighbor_second_order_percent", data=np.array(best_neighbor_type_history))
            
            for pos in track_positions:
                data_file.create_dataset(f"q_c_pos_{pos[0]}_{pos[1]}_final", data=np.array(q_history_agg[pos]['q_c']))
                data_file.create_dataset(f"q_d_pos_{pos[0]}_{pos[1]}_final", data=np.array(q_history_agg[pos]['q_d']))
            
            data_file.create_dataset("Sn_final", data=Sn_final)
            data_file.create_dataset("R_final", data=R_final)
            data_file.create_dataset("cluster_sizes", data=np.array(cluster_sizes))
            data_file.create_dataset("reputation_reward_ratio", data=np.array(reputation_reward_ratio))
            data_file.create_dataset("avg_reward_C_history", data=np.array(avg_reward_C_history))
            data_file.create_dataset("avg_reward_D_history", data=np.array(avg_reward_D_history))
            
            for num_d in range(6):
                 data_file.create_dataset(f"group_comp_d{num_d}_history", 
                                         data=np.array(group_composition_history[num_d]))

            for group_name, q_hist_dict in q_history_by_strategy.items():
                for key, value in q_hist_dict.items():
                    data_file.create_dataset(f"{group_name}_{key}_history", data=np.array(value))

            for key, value in avg_q_history.items():
                data_file.create_dataset(f"avg_{key}_history", data=np.array(value))

            data_file.create_dataset("rep_hist_final", data=rep_hist_final)
            data_file.create_dataset("rep_bins_final", data=rep_bins_final)

    def save_snapshot(self, iteration: int, R: np.ndarray, Sn: np.ndarray, R_min: float, R_max: float):
        """Saves a snapshot of the simulation state."""
        # Note: In the original code, snapshots were saved to the same HDF5 file incrementally.
        # To support that, we need to open in 'a' (append) mode if file exists, or handle it carefully.
        # However, the original code opens 'w' at the start and keeps it open? 
        # Actually, the original code opens 'w' inside run() and keeps it open for the whole duration.
        # Here we might need a different approach if we want to save snapshots incrementally without keeping the file handle open everywhere.
        # For now, let's assume we pass the open file handle or we append.
        
        # Re-opening in 'a' mode is safer for modularity.
        with h5py.File(self.filename, "a") as data_file:
             data_file.create_dataset(f"R_snapshot_{iteration}", data=R)
             rep_hist, rep_bins = np.histogram(R, bins=20, range=(R_min, R_max))
             data_file.create_dataset(f"rep_hist_{iteration}", data=rep_hist)
             data_file.create_dataset(f"rep_bins_{iteration}", data=rep_bins)
             data_file.create_dataset(f"Sn_snapshot_{iteration}", data=Sn)

