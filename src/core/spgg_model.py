
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy.ndimage import label
from typing import List, Dict, Tuple, Optional

from src.config import SimulationConfig
from src.core.state_strategies import StateProvider
from src.utils.math_utils import overlap5
from src.utils.data_utils import DataManager

class SPGG:
    """
    Spatial Public Goods Game (SPGG) Simulation Core.
    """
    def __init__(self, config: SimulationConfig, state_provider: StateProvider, folder: str = "."):
        self.config = config
        self.state_provider = state_provider
        self.folder = folder
        
        # Unpack commonly used config for easier access (optional, but keeps code similar to original)
        self.L = config.L
        self.iterations = config.iterations
        self.num_of_strategies = config.num_of_strategies
        
        # Initialize Random Seed
        # np.random.seed() # Removed to allow external seeding for reproducibility
        
        # Initialize Q-table: (L, L, num_states, num_actions)
        # Assuming 2 states and 2 actions for now as per original code
        self.q_table = np.random.uniform(low=-0.01, high=0.01, size=(self.L, self.L, 2, 2))
        
        # Initialize Reputation
        self.R = np.zeros((self.L, self.L))
        
        # Initialize Population
        self._Sn = np.random.randint(0, 2, size=(self.L, self.L))
        self._S = [(self._Sn == j).astype(int) for j in range(self.num_of_strategies)]
        
        # Cache for S, N, P calculations
        self.cache = {}
        
        # Tracking
        self.track_positions = [(self.L//2, self.L//2), (self.L//4, self.L//4), (3*self.L//4, 3*self.L//4)]
        self.q_history = {pos: {'q_c': [], 'q_d': []} for pos in self.track_positions}
        
        # History containers
        self.it_records = []
        self.epsilon_history = []
        self.rep_avg_history = []
        self.influence_counts = []
        self.best_neighbor_type_history = []
        self.group_composition_history = [[] for _ in range(6)]
        self.avg_q_history = {
            'q_s0_c': [], 'q_s0_d': [],
            'q_s1_c': [], 'q_s1_d': []
        }
        self.q_history_by_strategy = {
            'cooperators': {key: [] for key in self.avg_q_history},
            'defectors': {key: [] for key in self.avg_q_history}
        }
        
        # Normalization factors
        self.normlize_max = 4 * config.r
        self.normlize_min = config.r - 5
        
        # Snapshot iterations
        max_pow = int(np.floor(np.log10(self.iterations)))
        self.snapshot_iters = {1, 10, 100, 1000, 5000, 10000, 20000, 30000, 40000}

        # Current epsilon
        self.epsilon = config.epsilon

    def fun_args_id(self, *args):
        return hash(args)

    def S(self, group_offset=(0, 0), member_offset=(0, 0)):
        key = self.fun_args_id("S", group_offset, member_offset)
        if key in self.cache:
            return self.cache[key]
        result = self._S
        if group_offset != (0, 0):
            result = [np.roll(s, *(group_offset)) for s in result]
        if member_offset != (0, 0):
            result = [np.roll(s, *(member_offset)) for s in result]
        self.cache[key] = result
        return result

    def N(self, group_offset=(0,0), member_offset=(0,0)):
        key = self.fun_args_id("N", group_offset)
        if key in self.cache:
            return self.cache[key]
        S = self.S(group_offset=group_offset)
        result = [overlap5(s) for s in S]
        self.cache[key] = result
        return result

    def P_g_m(self, group_offset=(0,0), member_offset=(0,0)):
        key = self.fun_args_id("P_g_m", group_offset, member_offset)
        if key in self.cache:
            return self.cache[key]
        N = self.N(group_offset, member_offset)
        S = self.S(group_offset, member_offset)
        n = 5
        # P = (r*c*Nc/n - cost)*Sc + (r*c*Nc/n)*Sd
        # Assuming c=1, cost from config
        P = ((self.config.r * self.config.c * N[0] / n - self.config.cost) * S[0] +
             (self.config.r * self.config.c * N[0] / n) * S[1])
        self.cache[key] = P
        return P

    def P_AT_g_m(self, group_offset=(0,0), member_offset=(0,0)):
        return self.P_g_m(group_offset, member_offset)

    def update_reputation(self, actions):
        delta_R = np.where(actions == 0, self.config.rep_gain_C, -self.config.delta_R_D)
        self.R += delta_R
        self.R = np.clip(self.R, self.config.R_min, self.config.R_max)

    def run(self, filename: str):
        L = self.L
        data_manager = DataManager(filename)
        
        # Local aggregators for history
        coop_rate_history = []
        it_records_agg = []
        epsilon_history_agg = []
        rep_avg_history_agg = []
        q_history_agg = {pos: {'q_c': [], 'q_d': []} for pos in self.track_positions}
        influence_counts_agg = []
        switch_C_to_D_agg = []
        switch_D_to_C_agg = []
        neighbor_influence_percent_agg = []
        payoff_comp_history = []
        rep_comp_history = []
        best_neighbor_type_history = []
        reputation_reward_ratio_history = []
        avg_reward_C_history = []
        avg_reward_D_history = []
        
        snapshots_dir = os.path.join(self.folder, 'plots', 'snapshots')
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Setup colormap for snapshots
        colors = ["#eeeeee", "#111111"]
        cmap = mpl.colors.ListedColormap(colors, N=2)

        for i in range(1, self.iterations + 1):
            prev_S = self._Sn.copy()
            
            # 1. Calculate Payoff P
            P = ((self.P_AT_g_m() +
                  self.P_AT_g_m((1,0), (-1,0)) +
                  self.P_AT_g_m((-1,0), (1,0)) +
                  self.P_AT_g_m((1,1), (-1,1)) +
                  self.P_AT_g_m((-1,1), (1,1))) - self.normlize_min) / (self.normlize_max - self.normlize_min)
            
            # 2. Record Stats
            S = self.S()
            S_coop, S_def = S[0], S[1]
            coop_rate = np.sum(S_coop) / (L * L)
            coop_rate_history.append(coop_rate)
            
            record = (coop_rate,
                      np.sum(S_def) / (L * L),
                      P.sum(), np.mean(P),
                      np.mean(P[self._Sn == 0]) if np.any(S_coop) else 0,
                      np.mean(P[self._Sn == 1]) if np.any(S_def) else 0)
            it_records_agg.append(record)
            rep_avg_history_agg.append(np.mean(self.R))
            
            # Save Snapshots
            if i in self.snapshot_iters:
                data_manager.save_snapshot(i, self.R, self._Sn, self.config.R_min, self.config.R_max)
                
            # Stop condition
            if coop_rate == 0 or coop_rate == 1:
                break
                
            # 3. Q-Learning Update
            old_states = self.state_provider.get_state(self.R, self._Sn)
            
            explore = np.random.rand(L, L) < self.epsilon
            q_values = self.q_table[np.arange(L)[:, None], np.arange(L), old_states, :]
            greedy_actions = np.argmax(q_values, axis=2)
            random_actions = np.random.randint(0, 2, size=(L, L))
            actions = np.where(explore, random_actions, greedy_actions)
            
            # Update Reputation & Strategy
            self.update_reputation(actions)
            self._Sn = actions.copy()
            self._S = [(actions == j).astype(int) for j in range(self.num_of_strategies)]
            self.cache = {}
            
            # Switches
            switch_C_to_D_agg.append(np.sum((prev_S == 0) & (self._Sn == 1)))
            switch_D_to_C_agg.append(np.sum((prev_S == 1) & (self._Sn == 0)))
            
            # Rewards
            new_states = self.state_provider.get_state(self.R, self._Sn)
            rep_reward = np.where(actions == 0, 0.5, 0) # Fixed 0.5 for C, 0 for D
            
            payoff_comp_history.append(np.mean(self.config.reward_weight_payoff * P))
            rep_comp_history.append(np.mean(self.config.reward_weight_rep * rep_reward))
            
            rewards = self.config.reward_weight_payoff * P + self.config.reward_weight_rep * rep_reward
            
            # Reputation Reward Ratio
            S_coop_mask = (actions == 0)
            if np.any(S_coop_mask):
                rewards_coop = rewards[S_coop_mask]
                rep_reward_coop = rep_reward[S_coop_mask]
                total_reward_magnitude_coop = np.abs(rewards_coop) + 1e-9
                reputation_ratio_for_cooperators = (np.abs(self.config.reward_weight_rep * rep_reward_coop) / total_reward_magnitude_coop) * 100
                reputation_reward_ratio_history.append(np.mean(reputation_ratio_for_cooperators))
            else:
                reputation_reward_ratio_history.append(np.nan)
                
            S_def_mask = (actions == 1)
            avg_reward_C_history.append(np.mean(rewards[S_coop_mask]) if np.any(S_coop_mask) else 0)
            avg_reward_D_history.append(np.mean(rewards[S_def_mask]) if np.any(S_def_mask) else 0)
            
            # Q-Table Update (TD)
            max_next_q = np.max(self.q_table[np.arange(L)[:, None], np.arange(L), new_states, :], axis=2)
            idx = np.indices((L, L))
            q_current = self.q_table[idx[0], idx[1], old_states, actions]
            td_error = rewards + self.config.gamma * max_next_q - q_current
            self.q_table[idx[0], idx[1], old_states, actions] += self.config.alpha * td_error
            
            # Neighbor Influence (NI)
            if self.config.use_second_order:
                offsets = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (2, 0), (-2, 0), (0, 2), (0, -2),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                ]
            else:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                
            diffs = np.stack([np.roll(rewards, shift=off, axis=(0,1)) - rewards for off in offsets], axis=0)
            max_diff = np.max(diffs, axis=0)
            global_max = np.max(np.abs(diffs))
            lambda_nei = self.config.influence_factor * np.maximum(0, max_diff) / (global_max + self.config.lambda_epsilon)
            max_idx = np.argmax(diffs, axis=0)
            nbr_actions = np.stack([np.roll(actions, shift=off, axis=(0,1)) for off in offsets], axis=0)
            rows, cols = np.indices((L, L))
            a_star = nbr_actions[max_idx, rows, cols]
            delta_beh = np.where(a_star == actions, 1.0, -1.0)
            neighbor_update = lambda_nei * delta_beh
            self.q_table[idx[0], idx[1], old_states, actions] += neighbor_update
            
            # NI Stats
            pct = np.abs(neighbor_update) / (np.abs(self.config.alpha * td_error) + np.abs(neighbor_update) + 1e-8) * 100
            neighbor_influence_percent_agg.append(np.mean(pct))
            
            # Best Neighbor Type
            if self.config.use_second_order:
                is_first_order = [True]*4 + [False]*8
            else:
                is_first_order = [True]*4
            best_is_second_order = ~np.array(is_first_order)[max_idx]
            mask = max_diff > 0
            percentage_second_order = np.mean(best_is_second_order[mask]) * 100 if np.any(mask) else 0
            best_neighbor_type_history.append(percentage_second_order)
            
            # Epsilon Decay
            self.epsilon = max(self.epsilon * self.config.epsilon_decay, self.config.epsilon_min)
            epsilon_history_agg.append(self.epsilon)
            
            # Snapshot Plot
            if i in self.snapshot_iters or i == 5000:
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(self._Sn, cmap=cmap, interpolation='nearest')
                ax.set_title(f"Strategy at iter={i}")
                ax.axis('off')
                fig.savefig(os.path.join(snapshots_dir, f"snapshot_{i}.png"))
                plt.close(fig)
                
            # Detailed Q Stats
            self.avg_q_history['q_s0_c'].append(np.mean(self.q_table[:, :, 0, 0]))
            self.avg_q_history['q_s0_d'].append(np.mean(self.q_table[:, :, 0, 1]))
            self.avg_q_history['q_s1_c'].append(np.mean(self.q_table[:, :, 1, 0]))
            self.avg_q_history['q_s1_d'].append(np.mean(self.q_table[:, :, 1, 1]))
            
            coop_mask = (prev_S == 0)
            def_mask = (prev_S == 1)
            
            for state_idx, state_name in enumerate(['s0', 's1']):
                for action_idx, action_name in enumerate(['c', 'd']):
                    q_values_all = self.q_table[:, :, state_idx, action_idx]
                    avg_q_coop = np.mean(q_values_all[coop_mask]) if np.any(coop_mask) else np.nan
                    self.q_history_by_strategy['cooperators'][f'q_{state_name}_{action_name}'].append(avg_q_coop)
                    avg_q_def = np.mean(q_values_all[def_mask]) if np.any(def_mask) else np.nan
                    self.q_history_by_strategy['defectors'][f'q_{state_name}_{action_name}'].append(avg_q_def)
                    
            # Group Composition
            S_def = self._S[1]
            num_defectors_in_each_group = overlap5(S_def)
            total_groups = L * L
            for num_d in range(6):
                count = np.sum(num_defectors_in_each_group == num_d)
                percentage = (count / total_groups) * 100
                self.group_composition_history[num_d].append(percentage)
                
            # Track Positions
            for pos in self.track_positions:
                x, y = pos
                state = old_states[x, y]
                q_history_agg[pos]['q_c'].append(self.q_table[x, y, state, 0])
                q_history_agg[pos]['q_d'].append(self.q_table[x, y, state, 1])

        # Save Final Data
        rep_hist_final, rep_bins_final = np.histogram(self.R, bins=20, range=(self.config.R_min, self.config.R_max))
        coop_clusters, num_clusters = label(self._Sn == 0)
        cluster_sizes = [np.sum(coop_clusters == idx) for idx in range(1, num_clusters + 1)]
        
        data_manager.save_final_data(
            it_records=it_records_agg,
            epsilon_history=epsilon_history_agg,
            rep_avg_history=rep_avg_history_agg,
            coop_rate_history=coop_rate_history,
            influence_counts=influence_counts_agg,
            switch_C_to_D=switch_C_to_D_agg,
            switch_D_to_C=switch_D_to_C_agg,
            neighbor_influence_percent=neighbor_influence_percent_agg,
            payoff_comp_history=payoff_comp_history,
            rep_comp_history=rep_comp_history,
            best_neighbor_type_history=best_neighbor_type_history,
            q_history_agg=q_history_agg,
            Sn_final=self._Sn,
            R_final=self.R,
            cluster_sizes=cluster_sizes,
            reputation_reward_ratio=reputation_reward_ratio_history,
            avg_reward_C_history=avg_reward_C_history,
            avg_reward_D_history=avg_reward_D_history,
            group_composition_history=self.group_composition_history,
            q_history_by_strategy=self.q_history_by_strategy,
            avg_q_history=self.avg_q_history,
            rep_hist_final=rep_hist_final,
            rep_bins_final=rep_bins_final,
            track_positions=self.track_positions
        )
        
        S_coop = (self._Sn == 0).astype(int)
        S_def = (self._Sn == 1).astype(int)
        return (np.sum(S_coop) / (L * L), np.sum(S_def) / (L * L), np.mean(P))
