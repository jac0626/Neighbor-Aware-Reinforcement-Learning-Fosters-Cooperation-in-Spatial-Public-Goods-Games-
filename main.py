import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from itertools import product
import multiprocessing
from scipy.ndimage import label
import h5py

# Color map settings
colors = [(48, 83, 133), (218, 160, 90)]
colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]
cmap_mma = LinearSegmentedColormap.from_list("mma", colors, N=256)
colors = ["#eeeeee", "#111111"]
cmap = mpl.colors.ListedColormap(colors, N=2)
reputation_cmap = plt.cm.viridis

# Helper function to compute the sum of a position and its four neighbors
def overlap5(A):
    return A + np.roll(A, -1, axis=0) + np.roll(A, 1, axis=0) + np.roll(A, -1, axis=1) + np.roll(A, 1, axis=1)

# SPGG Model Definition
class SPGG:
    def __init__(self, r=2, c=1, cost=0.5, K=0.1, L=50, iterations=1000, num_of_strategies=2, 
                 population_type=0, S_in_one=None, alpha=0.1, gamma=0.9, epsilon=0.5, 
                 epsilon_decay=0.995, epsilon_min=0.01, influence_factor=1.0, 
                 use_second_order=True, lambda_epsilon=0.01, delta_R_C=1, delta_R_D=1, 
                 R_min=-10, R_max=10, reward_weight_payoff=1.0, rep_gain_C=0.5, **params):
        np.random.seed()
        all_params = dict(locals(), **params)
        del all_params['self'], all_params['params']
        self.params = all_params
        for key in self.params:
            setattr(self, key, self.params[key])
        
        self.reward_weight_rep = 1 - self.reward_weight_payoff
        self.q_table = np.random.uniform(low=-0.01, high=0.01, size=(L, L, 2, 2))
        self.R = np.zeros((L, L))
        
        self.cache = {}
        self._Sn = S_in_one
        self.create_population()
        
        self.track_positions = [(L//2, L//2), (L//4, L//4), (3*L//4, 3*L//4)]
        self.q_history = {pos: {'q_c': [], 'q_d': []} for pos in self.track_positions}
        
        self.it_records = []
        self.epsilon_history = []
        self.rep_avg_history = []
        self.influence_counts = []
        # Normalization factor
        self.normlize_max = 4*r
        self.normlize_min = r-5

        # Prepare snapshot iteration points: 1,10,100,...
        max_pow = int(np.floor(np.log10(self.iterations)))
        self.snapshot_iters = {10**k for k in range(max_pow+1)}  

    def create_population(self):
        L = self.L
        if self._Sn is None:
            self._Sn = np.random.randint(0, 2, size=(L, L))
        self._S = [(self._Sn == j).astype(int) for j in range(self.num_of_strategies)]
        return self._S

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
        # Raw payoff, normalized
        P = ((self.r * self.c * N[0] / n - self.cost) * S[0] +
             (self.r * self.c * N[0] / n) * S[1])
        self.cache[key] = P
        return P

    def P_AT_g_m(self, group_offset=(0,0), member_offset=(0,0)):
        return self.P_g_m(group_offset, member_offset)

    def get_reputation_state(self):
        L = self.L
        if self.use_second_order:
            offsets = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (2, 0), (-2, 0), (0, 2), (0, -2),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]
        else:
            offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        sum_rep = np.zeros((L, L))
        for dx, dy in offsets:
            sum_rep += np.roll(self.R, shift=(dx, dy), axis=(0, 1))
        avg_rep = sum_rep / len(offsets)
        return (avg_rep > 0).astype(int)

    def update_reputation(self, actions):
        delta_R = np.where(actions == 0, self.rep_gain_C, -self.delta_R_D)
        self.R += delta_R
        self.R = np.clip(self.R, self.R_min, self.R_max)

    def run(self, filename):
        L = self.L
        with h5py.File(filename, "w") as data_file:
            # For saving cooperation rate snapshots
            coop_rate_history = []

            it_records_agg = []
            epsilon_history_agg = []
            rep_avg_history_agg = []
            q_history_agg = {pos: {'q_c': [], 'q_d': []} for pos in self.track_positions}
            influence_counts_agg = []
            switch_C_to_D_agg = []
            switch_D_to_C_agg = []
            avg_Q_coop_agg = []
            avg_Q_def_agg = []
            neighbor_influence_percent_agg = []
            payoff_comp_history = []
            rep_comp_history    = []


            snapshots_dir = os.path.join(self.folder, 'plots', 'snapshots')
            os.makedirs(snapshots_dir, exist_ok=True)

            for i in range(1, self.iterations + 1):
                prev_S = self._Sn.copy()

                # 1. Calculate the normalized payoff matrix P
                P = ((self.P_AT_g_m() +
                     self.P_AT_g_m((1,0), (-1,0)) +
                     self.P_AT_g_m((-1,0), (1,0)) +
                     self.P_AT_g_m((1,1), (-1,1)) +
                     self.P_AT_g_m((-1,1), (1,1)))-self.normlize_min)/(self.normlize_max-self.normlize_min)
                self.P = P

                # 2. Record strategy distribution and cooperation rate
                S = self.S()
                S_coop, S_def = S[0], S[1]
                coop_rate = np.sum(S_coop) / (L*L)
                coop_rate_history.append(coop_rate)

                record = (coop_rate,
                          np.sum(S_def)/(L*L),
                          P.sum(), np.mean(P),
                          np.mean(P[self._Sn==0]) if np.any(S_coop) else 0,
                          np.mean(P[self._Sn==1]) if np.any(S_def) else 0)
                it_records_agg.append(record)

                # 3. Record average reputation
                rep_avg_history_agg.append(np.mean(self.R))

                # 4. Stop if all defect or all cooperate
                if coop_rate == 0 or coop_rate == 1:
                    break

                # 5. Q-learning & neighbor influence updates (same as original code)
                old_states = self.get_reputation_state()
                explore = np.random.rand(L, L) < self.epsilon
                q_values = self.q_table[np.arange(L)[:, None], np.arange(L), old_states, :]
                greedy_actions = np.argmax(q_values, axis=2)
                random_actions = np.random.randint(0, 2, size=(L, L))
                actions = np.where(explore, random_actions, greedy_actions)

                # Record Q-values at tracked points
                for pos in self.track_positions:
                    x, y = pos
                    state = old_states[x, y]
                    q_history_agg[pos]['q_c'].append(self.q_table[x, y, state, 0])
                    q_history_agg[pos]['q_d'].append(self.q_table[x, y, state, 1])

                # Update reputation, strategy, cache
                self.update_reputation(actions)
                self._Sn = actions.copy()
                self._S = [(actions == j).astype(int) for j in range(self.num_of_strategies)]
                self.cache = {}

                # Calculate number of switches
                switch_C_to_D_agg.append(np.sum((prev_S == 0) & (self._Sn == 1)))
                switch_D_to_C_agg.append(np.sum((prev_S == 1) & (self._Sn == 0)))

                # Calculate reward and update Q-table
                new_states = self.get_reputation_state()
                rep_reward = np.where(actions == 0, 0.5, 0)
                payoff_comp_history.append(np.mean(self.reward_weight_payoff * P))
                rep_comp_history.append(   np.mean(self.reward_weight_rep   * rep_reward))
                rewards = self.reward_weight_payoff * P + self.reward_weight_rep * rep_reward

                max_next_q = np.max(self.q_table[np.arange(L)[:, None], np.arange(L), new_states, :], axis=2)
                idx = np.indices((L, L))
                q_current = self.q_table[idx[0], idx[1], old_states, actions]
                td_error = rewards + self.gamma * max_next_q - q_current
                alpha_t = self.alpha
                self.q_table[idx[0], idx[1], old_states, actions] += alpha_t * td_error

                # Neighbor influence update (same as original code)
                if self.use_second_order:
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
                lambda_nei = self.influence_factor * np.maximum(0, max_diff) / (global_max + self.lambda_epsilon)
                max_idx = np.argmax(diffs, axis=0)
                nbr_actions = np.stack([np.roll(actions, shift=off, axis=(0,1)) for off in offsets], axis=0)
                rows, cols = np.indices((L, L))
                a_star = nbr_actions[max_idx, rows, cols]
                delta_beh = np.where(a_star == actions, 1.0, -1.0)
                neighbor_update = lambda_nei * delta_beh
                self.q_table[idx[0], idx[1], old_states, actions] += neighbor_update

                # Calculate neighbor influence percentage
                pct = np.abs(neighbor_update) / (np.abs(alpha_t * td_error) + np.abs(neighbor_update) + 1e-8) * 100
                neighbor_influence_percent_agg.append(np.mean(pct))

                # Decay epsilon
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history_agg.append(self.epsilon)

                # -- Save snapshots at 10^k iterations -- 
                if i in self.snapshot_iters or i==5000:
                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.imshow(self._Sn, cmap=cmap, interpolation='nearest')
                    ax.set_title(f"Strategy at iter={i}")
                    ax.axis('off')
                    fig.savefig(os.path.join(snapshots_dir, f"snapshot_{i}.png"))
                    plt.close(fig)

            # -- Save all datasets -- 
            data_file.create_dataset("it_records_final", data=np.array(it_records_agg))
            data_file.create_dataset("epsilon_history_final", data=np.array(epsilon_history_agg))
            data_file.create_dataset("rep_avg_history_final", data=np.array(rep_avg_history_agg))
            data_file.create_dataset("coop_rate_history", data=np.array(coop_rate_history))
            data_file.create_dataset("influence_counts", data=np.array(influence_counts_agg))
            data_file.create_dataset("switch_C_to_D", data=np.array(switch_C_to_D_agg))
            data_file.create_dataset("switch_D_to_C", data=np.array(switch_D_to_C_agg))
            data_file.create_dataset("neighbor_influence_percent", data=np.array(neighbor_influence_percent_agg))
            data_file.create_dataset("payoff_component_history", data=np.array(payoff_comp_history))
            data_file.create_dataset("rep_component_history", data=np.array(rep_comp_history))
            for pos in self.track_positions:
                data_file.create_dataset(f"q_c_pos_{pos[0]}_{pos[1]}_final", data=np.array(q_history_agg[pos]['q_c']))
                data_file.create_dataset(f"q_d_pos_{pos[0]}_{pos[1]}_final", data=np.array(q_history_agg[pos]['q_d']))
            data_file.create_dataset("Sn_final", data=self._Sn)
            data_file.create_dataset("R_final", data=self.R)
            
            coop_clusters, num_clusters = label(self._Sn == 0)
            cluster_sizes = [np.sum(coop_clusters == idx) for idx in range(1, num_clusters+1)]
            data_file.create_dataset("cluster_sizes", data=np.array(cluster_sizes))

        # Return final cooperation rate, etc.
        S_coop = (self._Sn == 0).astype(int)
        S_def   = (self._Sn == 1).astype(int)
        return (np.sum(S_coop)/(L*L), np.sum(S_def)/(L*L), np.mean(P))

# Parallel Experiment Function
def run_one_experiment(params):
    r_val, influence_factor, use_second_order, alpha_val, reward_weight_payoff, rep_gain_C = params
    folder_name = f"results_r{r_val}_inf{influence_factor}_order{use_second_order}_alpha{alpha_val}_rw{reward_weight_payoff:.2f}_rgC{rep_gain_C:.2f}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(os.path.join(folder_name, "configurations"))
        os.makedirs(os.path.join(folder_name, "reputations"))
        os.makedirs(os.path.join(folder_name, "plots"))
        os.makedirs(os.path.join(folder_name, "plots", "snapshots"))
        os.makedirs(os.path.join(folder_name, "data"))
    
    spgg = SPGG(r=r_val, c=1, cost=1, iterations=20000, L=100,
                num_of_strategies=2, K=0.1, population_type=0,
                alpha=alpha_val, gamma=0.9, epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.01,
                influence_factor=influence_factor, use_second_order=use_second_order, lambda_epsilon=0.01,
                delta_R_C=1, delta_R_D=1, R_min=-10, R_max=10,
                reward_weight_payoff=reward_weight_payoff, rep_gain_C=rep_gain_C)
    spgg.folder = folder_name
    filename = os.path.join(folder_name, "data", "experiment_data.h5")
    record = spgg.run(filename)
    
    final_coop_ratio, final_def_ratio, _ = record
    final_rep_mean = spgg.rep_avg_history[-1] if spgg.rep_avg_history else 0
    print(f"Done: r={r_val}, φ={influence_factor}, order={use_second_order}, α={alpha_val}, w_P={reward_weight_payoff}, ΔR_C={rep_gain_C}")
    return params, (final_coop_ratio, final_rep_mean)

# Plotting Function
def plot_from_data(folder_name):
    filename = os.path.join(folder_name, "data", "experiment_data.h5")
    with h5py.File(filename, "r") as data_file:
        coop_hist = data_file["coop_rate_history"][:]
        neighbor_pct = data_file["neighbor_influence_percent"][:]
        pay  = data_file["payoff_component_history"][:]
        rep  = data_file["rep_component_history"][:]
    it = np.arange(1,len(coop_hist)+1)
    plt.figure(figsize=(8,6))
    plt.plot(it, pay, label="w_P · P", linestyle='-')
    plt.plot(it, rep, label="w_R · ΔR", linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward Component")
    plt.title("Payoff vs Reputation Component over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder,"plots","payoff_vs_rep.png"))
    plt.close()
    steps = np.arange(1, len(coop_hist) + 1)

    # Cooperation rate evolution
    plt.figure(figsize=(8,6))
    plt.plot(steps, coop_hist, label="Coop Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Cooperation Rate")
    plt.title("Cooperation Rate Evolution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "plots", "coop_rate_evolution.png"))
    plt.close()

    # Neighbor influence %
    plt.figure(figsize=(8,6))
    plt.plot(steps, neighbor_pct, label="Neighbor Influence %", color='purple')
    plt.xlabel("Iteration")
    plt.ylabel("Percentage (%)")
    plt.title("Neighbor Influence on Q-table Updates")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "plots", "neighbor_influence_percent.png"))
    plt.close()

if __name__ == '__main__':
    r_list = [2,4]
    influence_factor_list = [0,0.5,1,1.5,2]
    use_second_order_list = [False,True]
    alpha_list = [0.8]
    reward_weight_payoff_list = [1]
    rep_gain_C_list = [1.0]

    param_combinations = list(product(
        r_list, influence_factor_list, use_second_order_list,
        alpha_list, reward_weight_payoff_list, rep_gain_C_list
    ))
    print(f"Total: {len(param_combinations)} combos, using {multiprocessing.cpu_count()} cores")

    with multiprocessing.Pool() as pool:
        results = pool.map(run_one_experiment, param_combinations)

    for params, _ in results:
        folder = f"results_r{params[0]}_inf{params[1]}_order{params[2]}_\"
                 f"alpha{params[3]}_rw{params[4]:.2f}_rgC{params[5]:.2f}"
        plot_from_data(folder)

    print("All done!")