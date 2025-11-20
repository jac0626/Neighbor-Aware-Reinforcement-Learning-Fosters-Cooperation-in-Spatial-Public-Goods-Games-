"""
Spatial Public Goods Game (SPGG) Model with Reinforcement Learning
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import label
import h5py
from .algorithms import create_algorithm, RLAlgorithm


# Color map settings
colors = [(48, 83, 133), (218, 160, 90)]
colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]
cmap_mma = LinearSegmentedColormap.from_list("mma", colors, N=256)
colors = ["#eeeeee", "#111111"]
cmap = mpl.colors.ListedColormap(colors, N=2)
reputation_cmap = plt.cm.viridis


def sum_position_and_neighbors(A):
    """
    Compute the sum of a position and its four neighbors (up, down, left, right)
    
    Parameters:
    -----------
    A : numpy.ndarray
        Input matrix
        
    Returns:
    --------
    numpy.ndarray : Matrix where each element is the sum of itself and its four neighbors
    """
    return A + np.roll(A, -1, axis=0) + np.roll(A, 1, axis=0) + np.roll(A, -1, axis=1) + np.roll(A, 1, axis=1)


class SPGG:
    """
    Spatial Public Goods Game with Reinforcement Learning and Reputation Mechanism
    
    This class implements the SPGG model with:
    - Multiple RL algorithms for strategy selection (Q-learning, SARSA, Expected SARSA, Double Q-learning)
    - Reputation-based state representation
    - Neighbor influence mechanism
    - Support for first-order and second-order neighbors
    """
    
    def __init__(self, r=2, c=1, cost=0.5, K=0.1, L=50, iterations=1000, 
                 num_of_strategies=2, population_type=0, S_in_one=None, 
                 alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.995, 
                 epsilon_min=0.01, influence_factor=1.0, use_second_order=True, 
                 lambda_epsilon=0.01, delta_R_C=1, delta_R_D=1, R_min=-10, 
                 R_max=10, reward_weight_payoff=1.0, rep_gain_C=0.5, 
                 state_representation='reputation', algorithm='qlearning', **params):
        """
        Initialize SPGG model
        
        Parameters:
        -----------
        r : float
            Multiplication factor for public goods
        c : float
            Contribution amount
        cost : float
            Cost of cooperation
        L : int
            Lattice size (L x L)
        iterations : int
            Number of iterations to run
        alpha : float
            Learning rate for Q-learning
        gamma : float
            Discount factor for Q-learning
        epsilon : float
            Initial exploration rate
        epsilon_decay : float
            Epsilon decay rate per iteration
        epsilon_min : float
            Minimum epsilon value
        influence_factor : float
            Strength of neighbor influence (kappa)
        use_second_order : bool
            Whether to use second-order neighbors (M=2) or only first-order (M=1)
        reward_weight_payoff : float
            Weight for payoff component in reward (w_P)
        rep_gain_C : float
            Reputation gain for cooperation (ΔR_C)
        state_representation : str
            State representation method: 'reputation' (default) or 'action'
            - 'reputation': State based on average reputation of neighbors
            - 'action': State based on previous action/strategy
        algorithm : str or RLAlgorithm
            RL algorithm to use: 'qlearning', 'sarsa', 'expected_sarsa', 'double_qlearning'
            or an RLAlgorithm instance
        """
        np.random.seed()
        
        # Store all parameters
        all_params = dict(locals(), **params)
        del all_params['self'], all_params['params']
        self.params = all_params
        for key in self.params:
            setattr(self, key, self.params[key])
        
        # Derived parameters
        self.reward_weight_rep = 1 - self.reward_weight_payoff
        
        # Initialize algorithm
        if isinstance(algorithm, str):
            self.algorithm = create_algorithm(
                algorithm, alpha, gamma, epsilon, epsilon_decay, epsilon_min, **params
            )
        elif isinstance(algorithm, RLAlgorithm):
            self.algorithm = algorithm
        else:
            raise ValueError(f"algorithm must be str or RLAlgorithm, got {type(algorithm)}")
        
        # Initialize Q-table and reputation
        self.q_table = np.random.uniform(low=-0.01, high=0.01, size=(L, L, 2, 2))
        
        # For Double Q-learning, initialize two Q-tables
        if hasattr(self.algorithm, 'initialize_q_tables'):
            self.algorithm.initialize_q_tables(self.q_table.shape)
            # Use combined Q-table as main Q-table
            self.q_table = self.algorithm.get_combined_q_table()
        
        self.R = np.zeros((L, L))
        
        # Caching and population
        self.cache = {}
        self._Sn = S_in_one
        self.create_population()
        
        # Tracking positions for Q-value history
        self.track_positions = [(L//2, L//2), (L//4, L//4), (3*L//4, 3*L//4)]
        self.q_history = {pos: {'q_c': [], 'q_d': []} for pos in self.track_positions}
        
        # History tracking
        self.it_records = []
        self.epsilon_history = []
        self.rep_avg_history = []
        self.influence_counts = []
        self.best_neighbor_type_history = []
        
        # Normalization factors
        self.normlize_max = 4 * r
        self.normlize_min = r - 5
        
        # Snapshot iteration points
        max_pow = int(np.floor(np.log10(self.iterations)))
        self.snapshot_iters = {1, 10, 100, 1000, 5000, 10000, 20000, 30000, 40000}
        
        # Folder for output (set externally)
        self.folder = None

    def create_population(self):
        """Initialize or create population strategy matrix"""
        L = self.L
        if self._Sn is None:
            self._Sn = np.random.randint(0, 2, size=(L, L))
        self._S = [(self._Sn == j).astype(int) for j in range(self.num_of_strategies)]
        return self._S

    def generate_cache_key(self, *args):
        """
        Generate a cache key from function arguments
        
        Parameters:
        -----------
        *args : tuple
            Function arguments to hash
            
        Returns:
        --------
        int : Hash value of the arguments
        """
        return hash(args)

    def get_strategy_matrix(self, group_offset=(0, 0), member_offset=(0, 0)):
        """
        Get strategy matrix with optional offsets for group and member positions
        
        Parameters:
        -----------
        group_offset : tuple, optional
            Offset for group position (default: (0, 0))
        member_offset : tuple, optional
            Offset for member position (default: (0, 0))
            
        Returns:
        --------
        list : List of strategy matrices, one for each strategy type
        """
        key = self.generate_cache_key("S", group_offset, member_offset)
        if key in self.cache:
            return self.cache[key]
        result = self._S
        if group_offset != (0, 0):
            result = [np.roll(s, *(group_offset)) for s in result]
        if member_offset != (0, 0):
            result = [np.roll(s, *(member_offset)) for s in result]
        self.cache[key] = result
        return result

    def count_cooperators_in_groups(self, group_offset=(0, 0), member_offset=(0, 0)):
        """
        Calculate the number of cooperators in each group (5-member groups)
        
        Parameters:
        -----------
        group_offset : tuple, optional
            Offset for group position (default: (0, 0))
        member_offset : tuple, optional
            Offset for member position (default: (0, 0))
            
        Returns:
        --------
        list : List of matrices, each containing the count of cooperators in each group
        """
        key = self.generate_cache_key("N", group_offset)
        if key in self.cache:
            return self.cache[key]
        S = self.get_strategy_matrix(group_offset=group_offset)
        result = [sum_position_and_neighbors(s) for s in S]
        self.cache[key] = result
        return result

    def calculate_payoff(self, group_offset=(0, 0), member_offset=(0, 0)):
        """
        Calculate payoff for each position in the lattice
        
        The payoff is calculated based on the public goods game:
        - Cooperators contribute c and pay cost
        - All players receive r * c * (number of cooperators) / group_size
        - Defectors receive the public good benefit without paying cost
        
        Parameters:
        -----------
        group_offset : tuple, optional
            Offset for group position (default: (0, 0))
        member_offset : tuple, optional
            Offset for member position (default: (0, 0))
            
        Returns:
        --------
        numpy.ndarray : Payoff matrix for each position
        """
        key = self.generate_cache_key("calculate_payoff", group_offset, member_offset)
        if key in self.cache:
            return self.cache[key]
        N = self.count_cooperators_in_groups(group_offset, member_offset)
        S = self.get_strategy_matrix(group_offset, member_offset)
        n = 5  # Group size (center + 4 neighbors)
        P = ((self.r * self.c * N[0] / n - self.cost) * S[0] +
             (self.r * self.c * N[0] / n) * S[1])
        self.cache[key] = P
        return P

    def calculate_aggregated_payoff(self, group_offset=(0, 0), member_offset=(0, 0)):
        """
        Calculate aggregated payoff (same as calculate_payoff in this implementation)
        
        This method is kept for compatibility and clarity. In this implementation,
        it simply calls calculate_payoff.
        
        Parameters:
        -----------
        group_offset : tuple, optional
            Offset for group position (default: (0, 0))
        member_offset : tuple, optional
            Offset for member position (default: (0, 0))
            
        Returns:
        --------
        numpy.ndarray : Payoff matrix for each position
        """
        return self.calculate_payoff(group_offset, member_offset)

    def get_state(self):
        """
        Get state representation based on configured method
        
        Returns:
        --------
        numpy.ndarray : Binary state matrix (0 or 1)
        """
        if self.state_representation == 'action':
            # State based on previous action/strategy
            return (self._Sn == 0).astype(int)
        elif self.state_representation == 'reputation':
            # State based on average reputation of neighbors
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
        else:
            raise ValueError(f"Unknown state_representation: {self.state_representation}. "
                           f"Must be 'reputation' or 'action'")
    
    def get_reputation_state(self):
        """
        Legacy method name - redirects to get_state()
        Kept for backward compatibility
        """
        return self.get_state()

    def update_reputation(self, actions):
        """Update reputation based on actions"""
        delta_R = np.where(actions == 0, self.rep_gain_C, -self.delta_R_D)
        self.R += delta_R
        self.R = np.clip(self.R, self.R_min, self.R_max)

    def run(self, filename):
        """
        Run the simulation and save results to HDF5 file
        
        Parameters:
        -----------
        filename : str
            Path to output HDF5 file
            
        Returns:
        --------
        tuple : (final_coop_rate, final_def_rate, mean_payoff)
        """
        L = self.L
        with h5py.File(filename, "w") as data_file:
            # Initialize history lists
            coop_rate_history = []
            it_records_agg = []
            epsilon_history_agg = []
            rep_avg_history_agg = []
            q_history_agg = {pos: {'q_c': [], 'q_d': []} for pos in self.track_positions}
            switch_C_to_D_agg = []
            switch_D_to_C_agg = []
            neighbor_influence_percent_agg = []
            payoff_comp_history = []
            rep_comp_history = []
            best_neighbor_type_history = []
            reputation_reward_ratio_history = []
            avg_reward_C_history = []
            avg_reward_D_history = []
            self.group_composition_history = [[] for _ in range(6)]
            self.avg_q_history = {
                'q_s0_c': [], 'q_s0_d': [],
                'q_s1_c': [], 'q_s1_d': []
            }
            self.q_history_by_strategy = {
                'cooperators': {key: [] for key in self.avg_q_history},
                'defectors': {key: [] for key in self.avg_q_history}
            }

            snapshots_dir = os.path.join(self.folder, 'plots', 'snapshots') if self.folder else 'snapshots'
            os.makedirs(snapshots_dir, exist_ok=True)

            for i in range(1, self.iterations + 1):
                prev_S = self._Sn.copy()
                
                # Compute normalized payoff matrix P
                # Aggregate payoffs from all overlapping groups (5 groups per position)
                P = ((self.calculate_aggregated_payoff() +
                      self.calculate_aggregated_payoff((1, 0), (-1, 0)) +
                      self.calculate_aggregated_payoff((-1, 0), (1, 0)) +
                      self.calculate_aggregated_payoff((1, 1), (-1, 1)) +
                      self.calculate_aggregated_payoff((-1, 1), (1, 1))) - self.normlize_min) / (self.normlize_max - self.normlize_min)
                self.P = P

                # Record strategy distribution and cooperation rate
                S = self.get_strategy_matrix()
                S_coop, S_def = S[0], S[1]
                coop_rate = np.sum(S_coop) / (L * L)
                coop_rate_history.append(coop_rate)

                record = (coop_rate,
                          np.sum(S_def) / (L * L),
                          P.sum(), np.mean(P),
                          np.mean(P[self._Sn == 0]) if np.any(S_coop) else 0,
                          np.mean(P[self._Sn == 1]) if np.any(S_def) else 0)
                it_records_agg.append(record)

                # Record average reputation
                rep_avg_history_agg.append(np.mean(self.R))

                # Save snapshots
                if i in self.snapshot_iters:
                    data_file.create_dataset(f"R_snapshot_{i}", data=self.R)
                    rep_hist, rep_bins = np.histogram(self.R, bins=20, range=(self.R_min, self.R_max))
                    data_file.create_dataset(f"rep_hist_{i}", data=rep_hist)
                    data_file.create_dataset(f"rep_bins_{i}", data=rep_bins)
                    data_file.create_dataset(f"Sn_snapshot_{i}", data=self._Sn)

                # Stop if all defect or all cooperate
                if coop_rate == 0 or coop_rate == 1:
                    break

                # RL algorithm action selection
                old_states = self.get_reputation_state()
                actions = self.algorithm.select_action(self.q_table, old_states, L)

                # Update reputation, strategy, and clear cache
                self.update_reputation(actions)
                self._Sn = actions.copy()
                self._S = [(actions == j).astype(int) for j in range(self.num_of_strategies)]
                self.cache = {}

                # Calculate strategy switches
                switch_C_to_D_agg.append(np.sum((prev_S == 0) & (self._Sn == 1)))
                switch_D_to_C_agg.append(np.sum((prev_S == 1) & (self._Sn == 0)))

                # Compute rewards and update Q-table
                new_states = self.get_reputation_state()
                rep_reward = np.where(actions == 0, 0.5, 0)
                payoff_comp_history.append(np.mean(self.reward_weight_payoff * P))
                rep_comp_history.append(np.mean(self.reward_weight_rep * rep_reward))
                rewards = self.reward_weight_payoff * P + self.reward_weight_rep * rep_reward

                # Update Q-table using algorithm-specific update rule
                # For SARSA, need to select next actions before updating
                from .algorithms import SARSA
                if isinstance(self.algorithm, SARSA):
                    # Select next actions for SARSA (on-policy)
                    next_actions = self.algorithm.select_action(self.q_table, new_states, L)
                    self.q_table = self.algorithm.update_q_table(
                        self.q_table, old_states, actions, rewards, new_states,
                        next_actions=next_actions
                    )
                else:
                    # For other algorithms (Q-learning, Expected SARSA, Double Q-learning)
                    self.q_table = self.algorithm.update_q_table(
                        self.q_table, old_states, actions, rewards, new_states
                    )
                
                # Calculate TD error for neighbor influence calculation
                idx = np.indices((L, L))
                q_current = self.q_table[idx[0], idx[1], old_states, actions]
                
                from .algorithms import SARSA, ExpectedSARSA
                if isinstance(self.algorithm, SARSA):
                    # SARSA uses next action's Q-value
                    next_actions = self.algorithm.select_action(self.q_table, new_states, L)
                    next_q = self.q_table[idx[0], idx[1], new_states, next_actions]
                    td_error = rewards + self.gamma * next_q - q_current
                elif isinstance(self.algorithm, ExpectedSARSA):
                    # Expected SARSA uses expected value
                    num_actions = self.q_table.shape[3]
                    next_q_values = self.q_table[np.arange(L)[:, None], np.arange(L), new_states, :]
                    greedy_next_actions = np.argmax(next_q_values, axis=2)
                    policy_probs = np.full((L, L, num_actions), self.algorithm.epsilon / num_actions)
                    rows, cols = np.indices((L, L))
                    policy_probs[rows, cols, greedy_next_actions] = (1 - self.algorithm.epsilon) + self.algorithm.epsilon / num_actions
                    expected_next_q = np.sum(policy_probs * next_q_values, axis=2)
                    td_error = rewards + self.gamma * expected_next_q - q_current
                else:
                    # Q-learning or Double Q-learning
                    max_next_q = np.max(self.q_table[np.arange(L)[:, None], np.arange(L), new_states, :], axis=2)
                    td_error = rewards + self.gamma * max_next_q - q_current
                
                alpha_t = self.alpha

                # Neighbor influence update
                if self.use_second_order:
                    offsets = [
                        (1, 0), (-1, 0), (0, 1), (0, -1),
                        (2, 0), (-2, 0), (0, 2), (0, -2),
                        (1, 1), (1, -1), (-1, 1), (-1, -1),
                    ]
                else:
                    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                diffs = np.stack([np.roll(rewards, shift=off, axis=(0, 1)) - rewards for off in offsets], axis=0)
                max_diff = np.max(diffs, axis=0)
                global_max = np.max(np.abs(diffs))
                lambda_nei = self.influence_factor * np.maximum(0, max_diff) / (global_max + self.lambda_epsilon)
                max_idx = np.argmax(diffs, axis=0)
                nbr_actions = np.stack([np.roll(actions, shift=off, axis=(0, 1)) for off in offsets], axis=0)
                rows, cols = np.indices((L, L))
                a_star = nbr_actions[max_idx, rows, cols]
                delta_beh = np.where(a_star == actions, 1.0, -1.0)
                neighbor_update = lambda_nei * delta_beh
                self.q_table[idx[0], idx[1], old_states, actions] += neighbor_update

                # Calculate neighbor influence percentage
                pct = np.abs(neighbor_update) / (np.abs(alpha_t * td_error) + np.abs(neighbor_update) + 1e-8) * 100
                neighbor_influence_percent_agg.append(np.mean(pct))

                # Track best neighbor type
                if self.use_second_order:
                    is_first_order = [True] * 4 + [False] * 8
                else:
                    is_first_order = [True] * 4
                best_is_second_order = ~np.array(is_first_order)[max_idx]
                mask = max_diff > 0
                if np.any(mask):
                    percentage_second_order = np.mean(best_is_second_order[mask]) * 100
                else:
                    percentage_second_order = 0
                best_neighbor_type_history.append(percentage_second_order)

                # Reputation reward ratio for cooperators
                S_coop_mask = (actions == 0)
                if np.any(S_coop_mask):
                    rewards_coop = rewards[S_coop_mask]
                    rep_reward_coop = rep_reward[S_coop_mask]
                    total_reward_magnitude_coop = np.abs(rewards_coop) + 1e-9
                    reputation_ratio_for_cooperators = (np.abs(self.reward_weight_rep * rep_reward_coop) / total_reward_magnitude_coop) * 100
                    avg_ratio_for_cooperators = np.mean(reputation_ratio_for_cooperators)
                    reputation_reward_ratio_history.append(avg_ratio_for_cooperators)
                else:
                    reputation_reward_ratio_history.append(np.nan)

                # Average rewards by strategy
                S_def_mask = (actions == 1)
                avg_reward_C = np.mean(rewards[S_coop_mask]) if np.any(S_coop_mask) else 0
                avg_reward_D = np.mean(rewards[S_def_mask]) if np.any(S_def_mask) else 0
                avg_reward_C_history.append(avg_reward_C)
                avg_reward_D_history.append(avg_reward_D)

                # Decay epsilon (use algorithm's epsilon)
                self.algorithm.decay_epsilon()
                self.epsilon = self.algorithm.epsilon  # Keep for compatibility
                epsilon_history_agg.append(self.epsilon)

                # Save strategy snapshot
                if i in self.snapshot_iters or i == 5000:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(self._Sn, cmap=cmap, interpolation='nearest')
                    ax.set_title(f"Strategy at iter={i}")
                    ax.axis('off')
                    fig.savefig(os.path.join(snapshots_dir, f"snapshot_{i}.png"))
                    plt.close(fig)

                # Record Q-values
                self.avg_q_history['q_s0_c'].append(np.mean(self.q_table[:, :, 0, 0]))
                self.avg_q_history['q_s0_d'].append(np.mean(self.q_table[:, :, 0, 1]))
                self.avg_q_history['q_s1_c'].append(np.mean(self.q_table[:, :, 1, 0]))
                self.avg_q_history['q_s1_d'].append(np.mean(self.q_table[:, :, 1, 1]))

                # Q-values by strategy
                coop_mask = (prev_S == 0)
                def_mask = (prev_S == 1)
                for state_idx, state_name in enumerate(['s0', 's1']):
                    for action_idx, action_name in enumerate(['c', 'd']):
                        q_values_all = self.q_table[:, :, state_idx, action_idx]
                        if np.any(coop_mask):
                            avg_q_coop = np.mean(q_values_all[coop_mask])
                        else:
                            avg_q_coop = np.nan
                        self.q_history_by_strategy['cooperators'][f'q_{state_name}_{action_name}'].append(avg_q_coop)
                        
                        if np.any(def_mask):
                            avg_q_def = np.mean(q_values_all[def_mask])
                        else:
                            avg_q_def = np.nan
                        self.q_history_by_strategy['defectors'][f'q_{state_name}_{action_name}'].append(avg_q_def)

                # Group composition
                S_def = self._S[1]
                num_defectors_in_each_group = sum_position_and_neighbors(S_def)
                total_groups = L * L
                for num_d in range(6):
                    count = np.sum(num_defectors_in_each_group == num_d)
                    percentage = (count / total_groups) * 100
                    self.group_composition_history[num_d].append(percentage)

            # Save all datasets
            data_file.create_dataset("it_records_final", data=np.array(it_records_agg))
            data_file.create_dataset("epsilon_history_final", data=np.array(epsilon_history_agg))
            data_file.create_dataset("rep_avg_history_final", data=np.array(rep_avg_history_agg))
            data_file.create_dataset("coop_rate_history", data=np.array(coop_rate_history))
            data_file.create_dataset("switch_C_to_D", data=np.array(switch_C_to_D_agg))
            data_file.create_dataset("switch_D_to_C", data=np.array(switch_D_to_C_agg))
            data_file.create_dataset("neighbor_influence_percent", data=np.array(neighbor_influence_percent_agg))
            data_file.create_dataset("payoff_component_history", data=np.array(payoff_comp_history))
            data_file.create_dataset("rep_component_history", data=np.array(rep_comp_history))
            data_file.create_dataset("best_neighbor_second_order_percent", data=np.array(best_neighbor_type_history))
            data_file.create_dataset("reputation_reward_ratio", data=np.array(reputation_reward_ratio_history))
            data_file.create_dataset("avg_reward_C_history", data=np.array(avg_reward_C_history))
            data_file.create_dataset("avg_reward_D_history", data=np.array(avg_reward_D_history))
            
            for num_d in range(6):
                data_file.create_dataset(f"group_comp_d{num_d}_history", 
                                       data=np.array(self.group_composition_history[num_d]))
            
            for group_name, q_hist_dict in self.q_history_by_strategy.items():
                for key, value in q_hist_dict.items():
                    data_file.create_dataset(f"{group_name}_{key}_history", data=np.array(value))
            
            for key, value in self.avg_q_history.items():
                data_file.create_dataset(f"avg_{key}_history", data=np.array(value))
            
            for pos in self.track_positions:
                data_file.create_dataset(f"q_c_pos_{pos[0]}_{pos[1]}_final", data=np.array(q_history_agg[pos]['q_c']))
                data_file.create_dataset(f"q_d_pos_{pos[0]}_{pos[1]}_final", data=np.array(q_history_agg[pos]['q_d']))
            
            data_file.create_dataset("Sn_final", data=self._Sn)
            data_file.create_dataset("R_final", data=self.R)
            
            rep_hist_final, rep_bins_final = np.histogram(self.R, bins=20, range=(self.R_min, self.R_max))
            data_file.create_dataset("rep_hist_final", data=rep_hist_final)
            data_file.create_dataset("rep_bins_final", data=rep_bins_final)
            
            coop_clusters, num_clusters = label(self._Sn == 0)
            cluster_sizes = [np.sum(coop_clusters == idx) for idx in range(1, num_clusters + 1)]
            data_file.create_dataset("cluster_sizes", data=np.array(cluster_sizes))

        S_coop = (self._Sn == 0).astype(int)
        S_def = (self._Sn == 1).astype(int)
        return (np.sum(S_coop) / (L * L), np.sum(S_def) / (L * L), np.mean(P))

