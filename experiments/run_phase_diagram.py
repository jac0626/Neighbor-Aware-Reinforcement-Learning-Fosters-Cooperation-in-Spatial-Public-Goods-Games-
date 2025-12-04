#!/usr/bin/env python3
"""
Phase Diagram Experiment Script.

Runs a chunk of the (lambda, r) parameter space for phase diagram generation.
Designed for parallel execution in GitHub Actions.
"""
import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import SimulationConfig
from src.core.spgg_model import SPGG
from src.core.state_strategies import ReputationStateProvider


def get_chunk_params(chunk_idx, total_points, chunk_size, param_min, param_max):
    """Get parameter values for a specific chunk."""
    all_values = np.linspace(param_min, param_max, total_points)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_points)
    return all_values[start_idx:end_idx]


def run_single_experiment(lam, r, iterations, L, output_dir):
    """Run a single experiment and return final cooperation rate."""
    import tempfile
    import h5py
    
    config = SimulationConfig(
        L=L,
        r=r,
        iterations=iterations,
        use_dqn=True,
        dqn_lambda=lam,
        influence_factor=0.0,
        use_soft_update=True,
        dqn_tau=0.005,
    )
    
    state_provider = ReputationStateProvider(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = os.path.join(tmpdir, f'exp_lam{lam:.3f}_r{r:.3f}')
        os.makedirs(folder, exist_ok=True)
        
        model = SPGG(config, state_provider, folder=folder)
        h5_path = os.path.join(folder, 'experiment_data.h5')
        model.run(h5_path)
        
        with h5py.File(h5_path, 'r') as f:
            coop_history = np.array(f['coop_rate_history'])
        
        # Return final cooperation rate (average of last 10%)
        final_window = max(1, len(coop_history) // 10)
        final_coop = np.mean(coop_history[-final_window:])
        
    return final_coop


def main():
    parser = argparse.ArgumentParser(description='Run phase diagram chunk')
    parser.add_argument('--lambda_chunk', type=int, required=True, help='Lambda chunk index')
    parser.add_argument('--r_chunk', type=int, required=True, help='R chunk index')
    parser.add_argument('--lambda_points', type=int, default=50, help='Total lambda points')
    parser.add_argument('--r_points', type=int, default=50, help='Total r points')
    parser.add_argument('--chunk_size', type=int, default=6, help='Points per chunk')
    parser.add_argument('--iterations', type=int, default=100000, help='Iterations per experiment')
    parser.add_argument('--L', type=int, default=100, help='Grid size')
    parser.add_argument('--output_dir', type=str, default='phase_results', help='Output directory')
    args = parser.parse_args()
    
    # Get parameter values for this chunk
    lambda_values = get_chunk_params(args.lambda_chunk, args.lambda_points, args.chunk_size, 0.0, 1.0)
    r_values = get_chunk_params(args.r_chunk, args.r_points, args.chunk_size, 1.0, 5.0)
    
    print(f"Phase Diagram Chunk: lambda_chunk={args.lambda_chunk}, r_chunk={args.r_chunk}")
    print(f"Lambda values: {lambda_values}")
    print(f"R values: {r_values}")
    print(f"Total experiments in this chunk: {len(lambda_values) * len(r_values)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Run experiments
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    n_workers = min(4, multiprocessing.cpu_count())
    print(f"Running with {n_workers} parallel workers")
    
    # Create list of all (lambda, r) pairs
    experiments = [(lam, r) for lam in lambda_values for r in r_values]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for lam, r in experiments:
            future = executor.submit(run_single_experiment, lam, r, args.iterations, args.L, args.output_dir)
            futures[future] = (lam, r)
        
        for future in as_completed(futures):
            lam, r = futures[future]
            try:
                final_coop = future.result()
                results.append({
                    'lambda': lam,
                    'r': r,
                    'final_coop': final_coop
                })
                print(f"  λ={lam:.3f}, r={r:.3f}: coop={final_coop:.4f}")
            except Exception as e:
                print(f"  λ={lam:.3f}, r={r:.3f}: ERROR - {e}")
                results.append({
                    'lambda': lam,
                    'r': r,
                    'final_coop': -1  # Error marker
                })
    
    # Save results
    output_file = os.path.join(args.output_dir, f'chunk_lam{args.lambda_chunk}_r{args.r_chunk}.json')
    with open(output_file, 'w') as f:
        json.dump({
            'lambda_chunk': args.lambda_chunk,
            'r_chunk': args.r_chunk,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
