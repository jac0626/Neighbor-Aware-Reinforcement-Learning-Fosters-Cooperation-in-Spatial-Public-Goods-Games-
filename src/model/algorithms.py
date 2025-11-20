"""
Reinforcement Learning Algorithms for SPGG
All algorithms are vectorized for efficient computation using NumPy
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class RLAlgorithm(ABC):
    """
    Base class for reinforcement learning algorithms
    All algorithms must implement the update_q_table method
    """
    
    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 epsilon_decay: float, epsilon_min: float, **kwargs):
        """
        Initialize RL algorithm
        
        Parameters:
        -----------
        alpha : float
            Learning rate
        gamma : float
            Discount factor
        epsilon : float
            Initial exploration rate
        epsilon_decay : float
            Epsilon decay rate per iteration
        epsilon_min : float
            Minimum epsilon value
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def decay_epsilon(self):
        """Decay epsilon value"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    @abstractmethod
    def update_q_table(self, q_table: np.ndarray, old_states: np.ndarray, 
                      actions: np.ndarray, rewards: np.ndarray, 
                      new_states: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update Q-table based on algorithm-specific update rule
        
        Parameters:
        -----------
        q_table : np.ndarray
            Q-table of shape (L, L, num_states, num_actions)
        old_states : np.ndarray
            Previous states, shape (L, L)
        actions : np.ndarray
            Actions taken, shape (L, L)
        rewards : np.ndarray
            Rewards received, shape (L, L)
        new_states : np.ndarray
            New states after taking actions, shape (L, L)
        **kwargs : dict
            Additional algorithm-specific parameters
            
        Returns:
        --------
        np.ndarray : Updated Q-table
        """
        pass
    
    @abstractmethod
    def select_action(self, q_table: np.ndarray, states: np.ndarray, 
                     L: int, **kwargs) -> np.ndarray:
        """
        Select actions based on epsilon-greedy policy
        
        Parameters:
        -----------
        q_table : np.ndarray
            Q-table of shape (L, L, num_states, num_actions)
        states : np.ndarray
            Current states, shape (L, L)
        L : int
            Lattice size
        **kwargs : dict
            Additional algorithm-specific parameters
            
        Returns:
        --------
        np.ndarray : Selected actions, shape (L, L)
        """
        pass


class QLearning(RLAlgorithm):
    """
    Q-learning algorithm (off-policy)
    Uses max Q-value of next state for update
    """
    
    def select_action(self, q_table: np.ndarray, states: np.ndarray, 
                     L: int, **kwargs) -> np.ndarray:
        """Select actions using epsilon-greedy policy"""
        explore = np.random.rand(L, L) < self.epsilon
        q_values = q_table[np.arange(L)[:, None], np.arange(L), states, :]
        greedy_actions = np.argmax(q_values, axis=2)
        random_actions = np.random.randint(0, 2, size=(L, L))
        actions = np.where(explore, random_actions, greedy_actions)
        return actions
    
    def update_q_table(self, q_table: np.ndarray, old_states: np.ndarray, 
                      actions: np.ndarray, rewards: np.ndarray, 
                      new_states: np.ndarray, **kwargs) -> np.ndarray:
        """
        Q-learning update: Q(s,a) += alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        """
        L = q_table.shape[0]
        idx = np.indices((L, L))
        
        # Get current Q-values
        q_current = q_table[idx[0], idx[1], old_states, actions]
        
        # Get max Q-value of next state (off-policy)
        max_next_q = np.max(q_table[np.arange(L)[:, None], np.arange(L), new_states, :], axis=2)
        
        # TD error
        td_error = rewards + self.gamma * max_next_q - q_current
        
        # Update Q-table
        q_table[idx[0], idx[1], old_states, actions] += self.alpha * td_error
        
        return q_table


class SARSA(RLAlgorithm):
    """
    SARSA algorithm (on-policy)
    Uses actual next action for update (requires next_actions parameter)
    """
    
    def select_action(self, q_table: np.ndarray, states: np.ndarray, 
                     L: int, **kwargs) -> np.ndarray:
        """Select actions using epsilon-greedy policy"""
        explore = np.random.rand(L, L) < self.epsilon
        q_values = q_table[np.arange(L)[:, None], np.arange(L), states, :]
        greedy_actions = np.argmax(q_values, axis=2)
        random_actions = np.random.randint(0, 2, size=(L, L))
        actions = np.where(explore, random_actions, greedy_actions)
        return actions
    
    def update_q_table(self, q_table: np.ndarray, old_states: np.ndarray, 
                      actions: np.ndarray, rewards: np.ndarray, 
                      new_states: np.ndarray, **kwargs) -> np.ndarray:
        """
        SARSA update: Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]
        Requires next_actions in kwargs
        """
        if 'next_actions' not in kwargs:
            raise ValueError("SARSA requires 'next_actions' parameter")
        
        next_actions = kwargs['next_actions']
        L = q_table.shape[0]
        idx = np.indices((L, L))
        
        # Get current Q-values
        q_current = q_table[idx[0], idx[1], old_states, actions]
        
        # Get Q-value of next state and next action (on-policy)
        next_q = q_table[idx[0], idx[1], new_states, next_actions]
        
        # TD error
        td_error = rewards + self.gamma * next_q - q_current
        
        # Update Q-table
        q_table[idx[0], idx[1], old_states, actions] += self.alpha * td_error
        
        return q_table


class ExpectedSARSA(RLAlgorithm):
    """
    Expected SARSA algorithm
    Uses expected Q-value over next state actions (weighted by policy)
    """
    
    def select_action(self, q_table: np.ndarray, states: np.ndarray, 
                     L: int, **kwargs) -> np.ndarray:
        """Select actions using epsilon-greedy policy"""
        explore = np.random.rand(L, L) < self.epsilon
        q_values = q_table[np.arange(L)[:, None], np.arange(L), states, :]
        greedy_actions = np.argmax(q_values, axis=2)
        random_actions = np.random.randint(0, 2, size=(L, L))
        actions = np.where(explore, random_actions, greedy_actions)
        return actions
    
    def update_q_table(self, q_table: np.ndarray, old_states: np.ndarray, 
                      actions: np.ndarray, rewards: np.ndarray, 
                      new_states: np.ndarray, **kwargs) -> np.ndarray:
        """
        Expected SARSA update: Q(s,a) += alpha * [r + gamma * E[Q(s',a')] - Q(s,a)]
        E[Q(s',a')] = sum_a' π(a'|s') * Q(s',a')
        where π is epsilon-greedy policy
        """
        L = q_table.shape[0]
        num_actions = q_table.shape[3]
        idx = np.indices((L, L))
        
        # Get current Q-values
        q_current = q_table[idx[0], idx[1], old_states, actions]
        
        # Get Q-values for next state
        next_q_values = q_table[np.arange(L)[:, None], np.arange(L), new_states, :]
        
        # Compute greedy actions for next state
        greedy_next_actions = np.argmax(next_q_values, axis=2)
        
        # Epsilon-greedy policy probabilities
        # For greedy action: (1 - epsilon) + epsilon / num_actions
        # For other actions: epsilon / num_actions
        policy_probs = np.full((L, L, num_actions), self.epsilon / num_actions)
        rows, cols = np.indices((L, L))
        policy_probs[rows, cols, greedy_next_actions] = (1 - self.epsilon) + self.epsilon / num_actions
        
        # Expected Q-value: sum over actions of π(a'|s') * Q(s',a')
        expected_next_q = np.sum(policy_probs * next_q_values, axis=2)
        
        # TD error
        td_error = rewards + self.gamma * expected_next_q - q_current
        
        # Update Q-table
        q_table[idx[0], idx[1], old_states, actions] += self.alpha * td_error
        
        return q_table


class DoubleQLearning(RLAlgorithm):
    """
    Double Q-learning algorithm
    Uses two Q-tables to reduce overestimation bias
    """
    
    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 epsilon_decay: float, epsilon_min: float, **kwargs):
        """Initialize Double Q-learning with two Q-tables"""
        super().__init__(alpha, gamma, epsilon, epsilon_decay, epsilon_min, **kwargs)
        self.q_table_1 = None
        self.q_table_2 = None
    
    def initialize_q_tables(self, shape: Tuple[int, ...]):
        """
        Initialize two Q-tables
        
        Parameters:
        -----------
        shape : tuple
            Shape of Q-table (L, L, num_states, num_actions)
        """
        self.q_table_1 = np.random.uniform(low=-0.01, high=0.01, size=shape)
        self.q_table_2 = np.random.uniform(low=-0.01, high=0.01, size=shape)
    
    def get_combined_q_table(self) -> np.ndarray:
        """Get combined Q-table (average of both) for action selection"""
        if self.q_table_1 is None or self.q_table_2 is None:
            raise ValueError("Q-tables not initialized. Call initialize_q_tables first.")
        return (self.q_table_1 + self.q_table_2) / 2
    
    def select_action(self, q_table: np.ndarray, states: np.ndarray, 
                     L: int, **kwargs) -> np.ndarray:
        """
        Select actions using combined Q-table
        Note: q_table parameter is ignored, uses internal combined Q-table
        """
        if self.q_table_1 is None or self.q_table_2 is None:
            # Fallback to provided q_table if not initialized
            explore = np.random.rand(L, L) < self.epsilon
            q_values = q_table[np.arange(L)[:, None], np.arange(L), states, :]
            greedy_actions = np.argmax(q_values, axis=2)
            random_actions = np.random.randint(0, 2, size=(L, L))
            actions = np.where(explore, random_actions, greedy_actions)
            return actions
        
        # Use combined Q-table
        combined_q = self.get_combined_q_table()
        explore = np.random.rand(L, L) < self.epsilon
        q_values = combined_q[np.arange(L)[:, None], np.arange(L), states, :]
        greedy_actions = np.argmax(q_values, axis=2)
        random_actions = np.random.randint(0, 2, size=(L, L))
        actions = np.where(explore, random_actions, greedy_actions)
        return actions
    
    def update_q_table(self, q_table: np.ndarray, old_states: np.ndarray, 
                      actions: np.ndarray, rewards: np.ndarray, 
                      new_states: np.ndarray, **kwargs) -> np.ndarray:
        """
        Double Q-learning update:
        Randomly choose which Q-table to update
        Use the other Q-table to evaluate next state action
        """
        if self.q_table_1 is None or self.q_table_2 is None:
            raise ValueError("Q-tables not initialized. Call initialize_q_tables first.")
        
        L = q_table.shape[0]
        idx = np.indices((L, L))
        
        # Randomly choose which Q-table to update (for each position)
        update_table_1 = np.random.rand(L, L) < 0.5
        
        # Get current Q-values from the table being updated
        q_current_1 = self.q_table_1[idx[0], idx[1], old_states, actions]
        q_current_2 = self.q_table_2[idx[0], idx[1], old_states, actions]
        
        # For positions updating Q1: use Q2 to evaluate next state
        # For positions updating Q2: use Q1 to evaluate next state
        q2_next_values = self.q_table_2[np.arange(L)[:, None], np.arange(L), new_states, :]
        q1_next_values = self.q_table_1[np.arange(L)[:, None], np.arange(L), new_states, :]
        
        # Get best actions from the evaluation table
        best_actions_q2 = np.argmax(q2_next_values, axis=2)
        best_actions_q1 = np.argmax(q1_next_values, axis=2)
        
        # Get Q-values using best actions from evaluation table
        rows, cols = np.indices((L, L))
        next_q_1 = q2_next_values[rows, cols, best_actions_q2]  # Q1 uses Q2's best action
        next_q_2 = q1_next_values[rows, cols, best_actions_q1]  # Q2 uses Q1's best action
        
        # TD errors
        td_error_1 = rewards + self.gamma * next_q_1 - q_current_1
        td_error_2 = rewards + self.gamma * next_q_2 - q_current_2
        
        # Update selected Q-table
        mask_1 = update_table_1
        mask_2 = ~update_table_1
        
        self.q_table_1[idx[0][mask_1], idx[1][mask_1], 
                       old_states[mask_1], actions[mask_1]] += self.alpha * td_error_1[mask_1]
        self.q_table_2[idx[0][mask_2], idx[1][mask_2], 
                       old_states[mask_2], actions[mask_2]] += self.alpha * td_error_2[mask_2]
        
        # Return combined Q-table for compatibility
        return self.get_combined_q_table()


def create_algorithm(algorithm_name: str, alpha: float, gamma: float, 
                     epsilon: float, epsilon_decay: float, epsilon_min: float, 
                     **kwargs) -> RLAlgorithm:
    """
    Factory function to create RL algorithm instances
    
    Parameters:
    -----------
    algorithm_name : str
        Name of algorithm: 'qlearning', 'sarsa', 'expected_sarsa', 'double_qlearning'
    alpha : float
        Learning rate
    gamma : float
        Discount factor
    epsilon : float
        Initial exploration rate
    epsilon_decay : float
        Epsilon decay rate
    epsilon_min : float
        Minimum epsilon
    **kwargs : dict
        Additional algorithm-specific parameters
        
    Returns:
    --------
    RLAlgorithm : Algorithm instance
    """
    algorithm_name = algorithm_name.lower()
    
    if algorithm_name == 'qlearning' or algorithm_name == 'q-learning':
        return QLearning(alpha, gamma, epsilon, epsilon_decay, epsilon_min, **kwargs)
    elif algorithm_name == 'sarsa':
        return SARSA(alpha, gamma, epsilon, epsilon_decay, epsilon_min, **kwargs)
    elif algorithm_name == 'expected_sarsa' or algorithm_name == 'expected-sarsa':
        return ExpectedSARSA(alpha, gamma, epsilon, epsilon_decay, epsilon_min, **kwargs)
    elif algorithm_name == 'double_qlearning' or algorithm_name == 'double-q-learning':
        return DoubleQLearning(alpha, gamma, epsilon, epsilon_decay, epsilon_min, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                        f"Supported: 'qlearning', 'sarsa', 'expected_sarsa', 'double_qlearning'")

