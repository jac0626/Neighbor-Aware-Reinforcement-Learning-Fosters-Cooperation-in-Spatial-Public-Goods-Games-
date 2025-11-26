
import os
import argparse
import multiprocessing
from itertools import product
from typing import Tuple, List

from src.config import SimulationConfig
from src.core.spgg_model import SPGG
from src.core.state_strategies import ReputationStateProvider, ActionStateProvider
from src.visualization.plot_figures import plot_from_data

def run_single_simulation(args: Tuple[SimulationConfig, str, str]) -> Tuple[str, Tuple[float, float, float]]:
    """
    Runs a single simulation instance.
    
    Args:
        args: Tuple containing (config, folder_name, state_type).
        
    Returns:
        Tuple of (folder_name, results).
    """
    config, folder_name, state_type = args
    
    # Create output directories
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(os.path.join(folder_name, "data"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "plots"), exist_ok=True)
    
    # Select State Provider
    if state_type == "reputation":
        state_provider = ReputationStateProvider(config)
    elif state_type == "action":
        state_provider = ActionStateProvider(config)
    else:
        raise ValueError(f"Unknown state type: {state_type}")
        
    # Initialize and Run SPGG
    spgg = SPGG(config, state_provider, folder=folder_name)
    filename = os.path.join(folder_name, "data", "experiment_data.h5")
    results = spgg.run(filename)
    
    print(f"Finished: {folder_name} | Coop Rate: {results[0]:.2f}")
    return folder_name, results

def main():
    parser = argparse.ArgumentParser(description="Run SPGG Experiments")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--state_type", type=str, default="reputation", choices=["reputation", "action"], help="Type of state definition")
    parser.add_argument("--output_dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--plot", action="store_true", help="Generate plots after simulation")
    
    # Override arguments
    parser.add_argument("--r", type=float, nargs="+", help="Override r values")
    parser.add_argument("--influence_factor", type=float, nargs="+", help="Override influence_factor values")
    parser.add_argument("--iterations", type=int, help="Override number of iterations")
    parser.add_argument("--dqn_lambda", type=float, nargs="+", help="Override dqn_lambda values")
    
    args = parser.parse_args()
    
    # Load Base Config
    if args.config:
        base_config = SimulationConfig.from_yaml(args.config)
    else:
        base_config = SimulationConfig()
        
    # Prepare Parameter Sweep
    r_list = args.r if args.r else [base_config.r]
    inf_list = args.influence_factor if args.influence_factor else [base_config.influence_factor]
    lambda_list = args.dqn_lambda if args.dqn_lambda else [base_config.dqn_lambda]
    
    # Generate Configurations
    tasks = []
    for r_val, inf_val, lam_val in product(r_list, inf_list, lambda_list):
        # Create a copy of config with updated values
        # Note: dataclass replace is cleaner but manual dict update works too
        config_dict = base_config.to_dict()
        config_dict['r'] = r_val
        config_dict['influence_factor'] = inf_val
        config_dict['dqn_lambda'] = lam_val
        if args.iterations:
            config_dict['iterations'] = args.iterations
        config = SimulationConfig.from_dict(config_dict)
        
        folder_name = os.path.join(args.output_dir, f"r{r_val}_inf{inf_val}_lam{lam_val}_{args.state_type}")
        tasks.append((config, folder_name, args.state_type))
        
    print(f"Running {len(tasks)} simulations...")
    
    if args.parallel:
        with multiprocessing.Pool() as pool:
            results = pool.map(run_single_simulation, tasks)
    else:
        results = [run_single_simulation(task) for task in tasks]
        
    print("All simulations completed.")
    
    if args.plot:
        print("Generating plots...")
        for folder, _ in results:
            plot_from_data(folder)
        print("Plotting completed.")

if __name__ == "__main__":
    main()
