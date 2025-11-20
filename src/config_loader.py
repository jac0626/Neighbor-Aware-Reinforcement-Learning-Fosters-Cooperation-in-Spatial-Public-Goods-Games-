"""
Configuration Loader Module
Handles loading and parsing configuration files
"""
import yaml
import os
from typing import Dict, Any, List, Tuple
from itertools import product


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str, optional
        Path to config file. If None, uses default config.
        
    Returns:
    --------
    dict : Configuration dictionary
    """
    if config_path is None:
        # Use default config
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        config_path = os.path.join(config_dir, 'default_config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def generate_param_combinations(config: Dict[str, Any], 
                               experiment_type: str = 'custom') -> List[Tuple]:
    """
    Generate parameter combinations from config
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    experiment_type : str
        Type of experiment ('figure_2_3_4', 'figure_6_7_8_9', 'custom')
        
    Returns:
    --------
    list : List of parameter tuples (r, kappa, use_second_order, alpha, w_P, rep_gain_C, state_representation)
    """
    model_config = config.get('model', {})
    rl_config = config.get('rl', {})
    rep_config = config.get('reputation', {})
    exp_config = config.get('experiments', {})
    
    # Default values
    base_alpha = rl_config.get('alpha', 0.8)
    base_rep_gain_C = rep_config.get('rep_gain_C', 1.0)
    state_representation = rl_config.get('state_representation', 'reputation')
    
    param_combinations = []
    
    if experiment_type == 'figure_2_3_4':
        fig_config = exp_config.get('figure_2_3_4', {})
        r = fig_config.get('r', 3.6)
        kappas = fig_config.get('kappas', [0.0, 0.5, 1.0, 1.5, 2.0])
        use_second_order_list = fig_config.get('use_second_order', [False, True])
        w_P = fig_config.get('reward_weight_payoff', 1.0)
        state_reps = fig_config.get('state_representation', [state_representation])
        if isinstance(state_reps, str):
            state_reps = [state_reps]
        
        for kappa, use_so, state_rep in product(kappas, use_second_order_list, state_reps):
            param_combinations.append((r, kappa, use_so, base_alpha, w_P, base_rep_gain_C, state_rep))
    
    elif experiment_type == 'figure_6_7_8_9':
        fig_config = exp_config.get('figure_6_7_8_9', {})
        r = fig_config.get('r', 3.0)
        use_second_order_list = fig_config.get('use_second_order', [False, True])
        state_reps = fig_config.get('state_representation', [state_representation])
        if isinstance(state_reps, str):
            state_reps = [state_reps]
        
        # Sole Reputation: kappa=0, w_P=0.95
        for use_so, state_rep in product(use_second_order_list, state_reps):
            param_combinations.append((r, 0.0, use_so, base_alpha, 0.95, base_rep_gain_C, state_rep))
        
        # Sole NI: kappa>0, w_P=1.0
        for use_so, state_rep in product(use_second_order_list, state_reps):
            param_combinations.append((r, 1.0, use_so, base_alpha, 1.0, base_rep_gain_C, state_rep))
        
        # Hybrid: kappa>0, w_P=0.95
        for use_so, state_rep in product(use_second_order_list, state_reps):
            param_combinations.append((r, 1.0, use_so, base_alpha, 0.95, base_rep_gain_C, state_rep))
    
    elif experiment_type == 'custom':
        custom_config = exp_config.get('custom', {})
        r_values = custom_config.get('r_values', [1.0, 2.0, 3.0, 4.0, 5.0])
        kappa_values = custom_config.get('kappa_values', [0.0, 0.5, 1.0, 1.5, 2.0])
        use_second_order_list = custom_config.get('use_second_order', [False, True])
        w_P_values = custom_config.get('reward_weight_payoff_values', [0.83, 0.95, 1.0])
        state_reps = custom_config.get('state_representation', [state_representation])
        if isinstance(state_reps, str):
            state_reps = [state_reps]
        
        for r, kappa, use_so, w_P, state_rep in product(r_values, kappa_values, 
                                             use_second_order_list, w_P_values, state_reps):
            param_combinations.append((r, kappa, use_so, base_alpha, w_P, base_rep_gain_C, state_rep))
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Remove duplicates
    param_combinations = list(set(param_combinations))
    return sorted(param_combinations)


def get_model_params(config: Dict[str, Any], **overrides) -> Dict[str, Any]:
    """
    Get model parameters from config with optional overrides
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    **overrides : dict
        Parameter overrides
        
    Returns:
    --------
    dict : Model parameters
    """
    model_config = config.get('model', {})
    rl_config = config.get('rl', {})
    ni_config = config.get('neighbor_influence', {})
    rep_config = config.get('reputation', {})
    reward_config = config.get('reward', {})
    
    params = {
        'r': model_config.get('r', 2.0),
        'c': model_config.get('c', 1.0),
        'cost': model_config.get('cost', 0.5),
        'K': model_config.get('K', 0.1),
        'L': model_config.get('L', 50),
        'iterations': model_config.get('iterations', 1000),
        'num_of_strategies': model_config.get('num_of_strategies', 2),
        'population_type': model_config.get('population_type', 0),
        'algorithm': rl_config.get('algorithm', 'qlearning'),
        'alpha': rl_config.get('alpha', 0.1),
        'gamma': rl_config.get('gamma', 0.9),
        'epsilon': rl_config.get('epsilon', 0.5),
        'epsilon_decay': rl_config.get('epsilon_decay', 0.995),
        'epsilon_min': rl_config.get('epsilon_min', 0.01),
        'state_representation': rl_config.get('state_representation', 'reputation'),
        'influence_factor': ni_config.get('influence_factor', 1.0),
        'use_second_order': ni_config.get('use_second_order', True),
        'lambda_epsilon': ni_config.get('lambda_epsilon', 0.01),
        'delta_R_C': rep_config.get('delta_R_C', 1.0),
        'delta_R_D': rep_config.get('delta_R_D', 1.0),
        'R_min': rep_config.get('R_min', -10),
        'R_max': rep_config.get('R_max', 10),
        'reward_weight_payoff': reward_config.get('reward_weight_payoff', 1.0),
        'rep_gain_C': rep_config.get('rep_gain_C', 0.5),
    }
    
    # Apply overrides
    params.update(overrides)
    
    return params

