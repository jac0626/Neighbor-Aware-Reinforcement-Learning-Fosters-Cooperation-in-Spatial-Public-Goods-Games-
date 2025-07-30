import os
import numpy as np
import multiprocessing
from scipy.ndimage import label
import h5py

# Helper function to compute the sum of a position and its four neighbors
def overlap5(A):
    return A + np.roll(A, -1, axis=0) + np.roll(A, 1, axis=0) + np.roll(A, -1, axis=1) + np.roll(A, 1, axis=1)

# SPGG Model Definition
class SPGG:
    def __init__(self, r=2, c=1, cost=1, K=0.1, L=100, iterations=20001, num_of_strategies=2,
                 population_type=0, S_in_one=None, alpha=0.8, gamma=0.9, epsilon=0.5,
                 epsilon_decay=0.99, epsilon_min=0.01, influence_factor=0.0,
                 use_second_order=False, lambda_epsilon=0.01, delta_R_C=1, delta_R_D=1,
                 R_min=-10, R_max=10, reward_weight_payoff=0.82, rep_gain_C=1.0, **params):
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

        # --- Initialize all history recorders ---
        self.avg_reward_by_category = {
            'C_in_1D': [], # Cooperators in 1D groups
            'D_in_2D': [], # Defectors in 2D groups
            'D_in_1D': [], # Defectors in 1D groups
        }
        
        # Normalization factors
        self.normlize_max = 4 * r
        self.normlize_min = r - 5


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

    def P_g_m(self, group_offset=(0,0), member_offset=(0,0)):
        key = self.fun_args_id("P_g_m", group_offset, member_offset)
        if key in self.cache:
            return self.cache[key]
        
        S_for_N = self.S(group_offset=group_offset)
        N_coop_in_group = overlap5(S_for_N[0])
        
        S = self.S(group_offset, member_offset)
        n = 5
        P = ((self.r * self.c * N_coop_in_group / n - self.cost) * S[0] +
             (self.r * self.c * N_coop_in_group / n) * S[1])
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
        else: # M=1, Von Neumann + self
            offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        
        sum_rep = np.zeros((L, L))
        for dx, dy in offsets:
            sum_rep += np.roll(self.R, shift=(dx, dy), axis=(0, 1))
        avg_rep = sum_rep / len(offsets)
        return (avg_rep > 0).astype(int)

    def update_reputation(self, actions):
        delta_R = np.where(actions == 0, self.delta_R_C, -self.delta_R_D)
        self.R += delta_R
        self.R = np.clip(self.R, self.R_min, self.R_max)

    def run(self, filename):
        L = self.L
        
        coop_rate_history = []
        
        for key in self.avg_reward_by_category: self.avg_reward_by_category[key] = []
        
        for i in range(1, self.iterations + 1):
            prev_S_matrix = self._Sn.copy()
            
            # --- Core Calculation ---
            self.cache = {}
            # Calculate the raw total payoff
            P_raw = (self.P_AT_g_m() + self.P_AT_g_m((1,0), (-1,0)) + self.P_AT_g_m((-1,0), (1,0)) +
                     self.P_AT_g_m((0,1), (0,-1)) + self.P_AT_g_m((0,-1), (0,1)))
            # Normalization
            P_normalized = (P_raw - self.normlize_min) / (self.normlize_max - self.normlize_min)
            
            # Agent decision-making
            old_states = self.get_reputation_state()
            explore = np.random.rand(L, L) < self.epsilon
            q_values = self.q_table[np.arange(L)[:, None], np.arange(L), old_states, :]
            greedy_actions = np.argmax(q_values, axis=2)
            random_actions = np.random.randint(0, 2, size=(L, L))
            actions_t = np.where(explore, random_actions, greedy_actions)
            
            # Calculate the composite reward
            rep_reward = np.where(actions_t == 0, 1.0, 0) # The reputation reward is +1/0 before normalization, and after normalization it is (1-0)/(1-0)=1 or (0-0)/(1-0)=0
                                                         # There is an error in formula (6) of the paper, it should be (ΔU_i(t) - ΔU_Tmin)/(ΔU_Tmax - ΔU_Tmin)
                                                         # Cooperate: (1 - (-1))/(1 - (-1)) = 1, Defect: (-1 - (-1))/(1 - (-1)) = 0
                                                         # Assuming the reputation change in the paper's model is only +1/-1 -> after normalization it is +1/0
            rewards = self.reward_weight_payoff * P_normalized + self.reward_weight_rep * rep_reward

            # --- (New) Track average reward by category ---
            # Use the strategy and group composition at time t to define categories and record the reward at time t
            current_S_matrix = actions_t
            S_def_t = (current_S_matrix == 1).astype(int)
            num_defectors_in_group_t = overlap5(S_def_t)

            coop_mask_t = (current_S_matrix == 0)
            def_mask_t = (current_S_matrix == 1)
            group_1D_mask_t = (num_defectors_in_group_t == 1)
            group_2D_mask_t = (num_defectors_in_group_t == 2)

            # C in 1D
            C_in_1D_mask = coop_mask_t & group_1D_mask_t
            self.avg_reward_by_category['C_in_1D'].append(np.mean(rewards[C_in_1D_mask]) if np.any(C_in_1D_mask) else np.nan)
            
            # D in 2D
            D_in_2D_mask = def_mask_t & group_2D_mask_t
            self.avg_reward_by_category['D_in_2D'].append(np.mean(rewards[D_in_2D_mask]) if np.any(D_in_2D_mask) else np.nan)

            # D in 1D
            D_in_1D_mask = def_mask_t & group_1D_mask_t
            self.avg_reward_by_category['D_in_1D'].append(np.mean(rewards[D_in_1D_mask]) if np.any(D_in_1D_mask) else np.nan)


            # Update Q-Table
            self.update_reputation(actions_t) # Reputation is updated before the Q-table, affecting new_states
            new_states = self.get_reputation_state()
            max_next_q = np.max(self.q_table[np.arange(L)[:, None], np.arange(L), new_states, :], axis=2)
            idx = np.indices((L, L))
            q_current = self.q_table[idx[0], idx[1], old_states, actions_t]
            td_error = rewards + self.gamma * max_next_q - q_current
            self.q_table[idx[0], idx[1], old_states, actions_t] += self.alpha * td_error
            
            # Update the strategy matrix for the next round
            self._Sn = actions_t.copy()
            self._S = [(self._Sn == j).astype(int) for j in range(self.num_of_strategies)]

            # Epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            # Record cooperation rate
            coop_rate = np.sum(self._S[0]) / (L*L)
            coop_rate_history.append(coop_rate)
            
            if coop_rate == 0 or coop_rate == 1:
                break
        
        # --- After the loop, save all data at once ---
        with h5py.File(filename, "w") as f:
            f.create_dataset("coop_rate_history", data=np.array(coop_rate_history))
            for key, value in self.avg_reward_by_category.items():
                f.create_dataset(f"avg_reward_{key}_history", data=np.array(value))

        return (coop_rate_history[-1] if coop_rate_history else 0, 0, 0)


# Parallel Experiment Function
def run_one_experiment(params):
    r_val, influence_factor, use_second_order, alpha_val, reward_weight_payoff, rep_gain_C = params
    folder_name = f"results_r{r_val:.1f}_inf{influence_factor:.1f}_order{use_second_order}_alpha{alpha_val}_rw{reward_weight_payoff:.2f}_rgC{rep_gain_C:.2f}"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(os.path.join(folder_name, "data"), exist_ok=True)

    spgg = SPGG(r=r_val, c=1, cost=1, iterations=20001, L=100,
                alpha=alpha_val, reward_weight_payoff=reward_weight_payoff, 
                use_second_order=use_second_order, influence_factor=influence_factor)
    
    filename = os.path.join(folder_name, "data", "experiment_data.h5")
    spgg.run(filename)
    
    print(f"Done: r={r_val}, κ={influence_factor}, M={use_second_order}, wP={reward_weight_payoff}")
    return True


if __name__ == '__main__':
    all_params_to_run = set()
    
    BASE_ALPHA = 0.8
    BASE_REP_GAIN_C = 1.00

    r_values = [1.0, 2.0, 3.0]
    for r in r_values:
        all_params_to_run.add((r, 0.0, False, BASE_ALPHA, 0.83, BASE_REP_GAIN_C))

    unique_params_list = sorted(list(all_params_to_run))

    print("="*60)
    print("Preparing to run simulations for reward comparison.")
    print(f"Total unique parameter combinations: {len(unique_params_list)}")
    for params in unique_params_list:
        print(f"  - r={params[0]:.1f}, kappa={params[1]:.1f}, M={params[2]}, wP={params[4]:.2f}")
    print("="*60)

    from tqdm import tqdm
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(run_one_experiment, unique_params_list), total=len(unique_params_list)))

    print("\nAll simulations are complete.")