#!/usr/bin/env python3
"""
Quick test script to compare Hard Update vs Soft Update for DQN.
Runs a single experiment at r=4.0 with lambda=1.0 for both configurations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SimulationConfig
from src.core.spgg_model import SPGG
from src.core.state_strategies import ReputationStateProvider

def run_comparison():
    """Run comparison between hard update and soft update."""
    
    # Base configuration
    base_config_dict = {
        'L': 50,  # Smaller grid for faster testing
        'iterations': 20000,  # Fewer iterations
        'r': 4.0,
        'c': 1.0,
        'cost': 1.0,
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon': 0.5,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'influence_factor': 0.0,  # Disable neighbor influence
        'use_dqn': True,
        'dqn_lambda': 1.0,  # Pure DQN
        'dqn_lr': 0.001,
        'dqn_update_freq': 10,
        'dqn_target_update_freq': 100,
        'dqn_tau': 0.005,
    }
    
    results = {}
    
    for update_type in ['hard', 'soft']:
        print(f"\n{'='*60}")
        print(f"Running with {update_type.upper()} update...")
        print(f"{'='*60}")
        
        # Create config
        config_dict = base_config_dict.copy()
        config_dict['use_soft_update'] = (update_type == 'soft')
        config = SimulationConfig.from_dict(config_dict)
        
        # Create folder
        folder = f"test_results/{update_type}_update"
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "data"), exist_ok=True)
        
        # Run simulation
        state_provider = ReputationStateProvider(config)
        spgg = SPGG(config, state_provider, folder=folder)
        filename = os.path.join(folder, "data", "experiment_data.h5")
        
        final_results = spgg.run(filename)
        
        # Load cooperation history
        import h5py
        with h5py.File(filename, 'r') as f:
            coop_history = f['coop_rate_history'][:]
        
        results[update_type] = {
            'coop_history': coop_history,
            'final_coop': final_results[0],
        }
        
        print(f"Final cooperation rate: {final_results[0]:.2%}")
        print(f"Mean (last 1000): {np.mean(coop_history[-1000:]):.2%}")
        print(f"Std (last 1000): {np.std(coop_history[-1000:]):.4f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full evolution
    ax1.plot(results['hard']['coop_history'], label='Hard Update', color='red', alpha=0.7, linewidth=1.5)
    ax1.plot(results['soft']['coop_history'], label='Soft Update', color='blue', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Cooperation Rate', fontsize=12)
    ax1.set_title('Full Evolution: Hard vs Soft Update (Pure DQN, r=4.0)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Last 10k iterations (zoom in)
    start_idx = max(0, len(results['hard']['coop_history']) - 10000)
    ax2.plot(range(start_idx, len(results['hard']['coop_history'])), 
             results['hard']['coop_history'][start_idx:], 
             label='Hard Update', color='red', alpha=0.7, linewidth=1.5)
    ax2.plot(range(start_idx, len(results['soft']['coop_history'])), 
             results['soft']['coop_history'][start_idx:], 
             label='Soft Update', color='blue', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Cooperation Rate', fontsize=12)
    ax2.set_title('Last 10k Iterations: Stability Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Add stats text
    hard_mean = np.mean(results['hard']['coop_history'][-1000:])
    hard_std = np.std(results['hard']['coop_history'][-1000:])
    soft_mean = np.mean(results['soft']['coop_history'][-1000:])
    soft_std = np.std(results['soft']['coop_history'][-1000:])
    
    stats_text = f"Stats (Last 1000 iter):\n"
    stats_text += f"Hard: μ={hard_mean:.2%}, σ={hard_std:.4f}\n"
    stats_text += f"Soft: μ={soft_mean:.2%}, σ={soft_std:.4f}\n"
    stats_text += f"Stability gain: {(hard_std/soft_std - 1)*100:.1f}%"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('test_results/hard_vs_soft_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: test_results/hard_vs_soft_comparison.png")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Hard Update: Mean={hard_mean:.2%}, Std={hard_std:.4f}")
    print(f"Soft Update: Mean={soft_mean:.2%}, Std={soft_std:.4f}")
    if soft_std < hard_std:
        print(f"✅ Soft update is {(hard_std/soft_std - 1)*100:.1f}% MORE STABLE!")
    else:
        print(f"⚠️ Soft update is {(soft_std/hard_std - 1)*100:.1f}% LESS stable")
    
    return results

if __name__ == "__main__":
    results = run_comparison()
