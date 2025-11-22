"""
Experiment Runner Module
Handles running single and batch experiments
"""
import os
import multiprocessing
from typing import Dict, List, Tuple, Any
from ..model import SPGG


def get_folder_name(r: float, kappa: float, use_second_order: bool, 
                   alpha: float, reward_weight_payoff: float, rep_gain_C: float,
                   state_representation: str = 'reputation', 
                   algorithm: str = 'qlearning') -> str:
    """
    Generate folder name from parameters
    
    Parameters:
    -----------
    r : float
        Multiplication factor
    kappa : float
        Influence factor
    use_second_order : bool
        Whether to use second-order neighbors
    alpha : float
        Learning rate
    reward_weight_payoff : float
        Payoff weight
    rep_gain_C : float
        Reputation gain for cooperation
    state_representation : str
        State representation method ('reputation' or 'action')
    algorithm : str
        RL algorithm name
        
    Returns:
    --------
    str : Folder name
    """
    order_str = str(use_second_order)
    state_suffix = "_action" if state_representation == 'action' else ""
    algo_suffix = f"_{algorithm}" if algorithm != 'qlearning' else ""
    return (f"results_r{r}_inf{kappa}_order{order_str}_alpha{alpha}_"
            f"rw{reward_weight_payoff:.2f}_rgC{rep_gain_C:.2f}{state_suffix}{algo_suffix}")


def run_one_experiment(params: Tuple) -> Tuple[Tuple, Tuple[float, float]]:
    """
    Run a single experiment with given parameters
    
    Parameters:
    -----------
    params : tuple
        (r, kappa, use_second_order, alpha, reward_weight_payoff, rep_gain_C, state_representation, algorithm)
        or (r, kappa, use_second_order, alpha, reward_weight_payoff, rep_gain_C, state_representation)
        or (r, kappa, use_second_order, alpha, reward_weight_payoff, rep_gain_C) for backward compatibility
        
    Returns:
    --------
    tuple : (params, (final_coop_ratio, final_rep_mean))
    """
    # Handle different parameter formats for backward compatibility
    if len(params) == 8:
        r_val, influence_factor, use_second_order, alpha_val, reward_weight_payoff, rep_gain_C, state_representation, algorithm = params
    elif len(params) == 7:
        r_val, influence_factor, use_second_order, alpha_val, reward_weight_payoff, rep_gain_C, state_representation = params
        algorithm = 'qlearning'  # Default algorithm
    else:
        r_val, influence_factor, use_second_order, alpha_val, reward_weight_payoff, rep_gain_C = params
        state_representation = 'reputation'  # Default for backward compatibility
        algorithm = 'qlearning'  # Default algorithm
    
    folder_name = get_folder_name(r_val, influence_factor, use_second_order, 
                                  alpha_val, reward_weight_payoff, rep_gain_C, 
                                  state_representation, algorithm)
    
    # Create directory structure
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(os.path.join(folder_name, "configurations"))
        os.makedirs(os.path.join(folder_name, "reputations"))
        os.makedirs(os.path.join(folder_name, "plots"))
        os.makedirs(os.path.join(folder_name, "plots", "snapshots"))
        os.makedirs(os.path.join(folder_name, "data"))
    
    # Create and run SPGG model
    spgg = SPGG(
        r=r_val, c=1, cost=1, iterations=100001, L=100,
        num_of_strategies=2, K=0.1, population_type=0,
        alpha=alpha_val, gamma=0.9, epsilon=0.5, 
        epsilon_decay=0.99, epsilon_min=0.01,
        influence_factor=influence_factor, 
        use_second_order=use_second_order, 
        lambda_epsilon=0.01,
        delta_R_C=1, delta_R_D=1, R_min=-10, R_max=10,
        reward_weight_payoff=reward_weight_payoff, 
        rep_gain_C=rep_gain_C,
        state_representation=state_representation,
        algorithm=algorithm
    )
    spgg.folder = folder_name
    
    filename = os.path.join(folder_name, "data", "experiment_data.h5")
    record = spgg.run(filename)
    
    final_coop_ratio, final_def_ratio, _ = record
    final_rep_mean = spgg.rep_avg_history[-1] if spgg.rep_avg_history else 0
    
    state_label = "action" if state_representation == 'action' else "rep"
    print(f"Done: r={r_val}, κ={influence_factor}, M={2 if use_second_order else 1}, "
          f"α={alpha_val}, w_P={reward_weight_payoff}, ΔR_C={rep_gain_C}, state={state_label}, algo={algorithm}")
    
    return params, (final_coop_ratio, final_rep_mean)


def run_experiments(param_combinations: List[Tuple], 
                   num_processes: int = None,
                   use_progress_bar: bool = True) -> List[Tuple]:
    """
    Run multiple experiments in parallel
    
    Parameters:
    -----------
    param_combinations : list of tuples
        List of parameter tuples to run
    num_processes : int, optional
        Number of processes to use (default: CPU count)
    use_progress_bar : bool
        Whether to show progress bar
        
    Returns:
    --------
    list : List of (params, results) tuples
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    if use_progress_bar:
        try:
            from tqdm import tqdm
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = []
                with tqdm(total=len(param_combinations), desc="Running simulations") as pbar:
                    for result in pool.imap_unordered(run_one_experiment, param_combinations):
                        results.append(result)
                        pbar.update()
        except ImportError:
            print("tqdm not available, running without progress bar")
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(run_one_experiment, param_combinations)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(run_one_experiment, param_combinations)
    
    return results

