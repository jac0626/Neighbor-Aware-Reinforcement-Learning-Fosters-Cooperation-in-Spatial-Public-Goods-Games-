
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import json
import os

@dataclass
class SimulationConfig:
    """
    Configuration for the Spatial Public Goods Game (SPGG) simulation.
    """
    # Grid dimensions
    L: int = 50
    
    # Game parameters
    r: float = 2.0
    c: float = 1.0
    cost: float = 0.5
    
    # Simulation parameters
    iterations: int = 1000
    num_of_strategies: int = 2
    population_type: int = 0
    
    # Q-learning parameters
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.5
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    K: float = 0.1 # Temperature for softmax (if used) or other scaling
    
    # Neighbor Influence (NI) parameters
    influence_factor: float = 1.0
    use_second_order: bool = True
    lambda_epsilon: float = 0.01
    
    # Reputation parameters
    delta_R_C: float = 1.0
    delta_R_D: float = 1.0
    R_min: float = -10.0
    R_max: float = 10.0
    rep_gain_C: float = 0.5
    
    # Reward weights
    reward_weight_payoff: float = 1.0
    
    # Derived parameters (not typically set by user, but can be)
    reward_weight_rep: float = field(init=False)
    
    # Hybrid Dual-Brain (DQN) parameters
    use_dqn: bool = False             # Whether to enable hybrid dual-brain architecture
    dqn_lambda: float = 0.5           # Mixing coefficient (0=pure Q-table, 1=pure DQN)
    dqn_lr: float = 1e-3              # DQN learning rate
    dqn_gamma: float = 0.9            # DQN discount factor
    dqn_hidden_dim: int = 64          # Hidden layer dimension
    dqn_buffer_size: int = 100000     # Replay buffer size
    dqn_batch_size: int = 64          # Training batch size
    dqn_update_freq: int = 10         # Network training frequency (every N rounds)
    dqn_input_dim: int = 4            # Continuous state vector dimension (default 4: self_rep, nei_rep, nei_coop, self_payoff)

    def __post_init__(self):
        self.reward_weight_rep = 1.0 - self.reward_weight_payoff

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """Creates a SimulationConfig from a dictionary, filtering unknown keys."""
        known_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'SimulationConfig':
        """Loads configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, file_path: str) -> 'SimulationConfig':
        """Loads configuration from a JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
