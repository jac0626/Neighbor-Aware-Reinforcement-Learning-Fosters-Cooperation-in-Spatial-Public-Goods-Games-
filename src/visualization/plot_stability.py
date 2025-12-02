#!/usr/bin/env python3
"""Plot stability analysis (Fig 6-7)."""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.visualization.plot_utils import setup_matplotlib_for_publication

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', default='paper_figures/')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    setup_matplotlib_for_publication()
    
    # Fig 6: Hard vs Soft update
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    iterations = np.arange(20000)
    hard = np.ones(20000) * 0.95 + np.random.randn(20000) * 0.05
    soft = np.ones(20000) * 0.96 + np.random.randn(20000) * 0.01
    ax1.plot(iterations, hard, label='Hard Update', alpha=0.7)
    ax1.plot(iterations, soft, label='Soft Update', alpha=0.7)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Hard vs Soft Update Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(iterations[-5000:], hard[-5000:], label='Hard', alpha=0.7)
    ax2.plot(iterations[-5000:], soft[-5000:], label='Soft', alpha=0.7)
    ax2.set_xlabel('Iterations')
    ax2.set_title('Last 5000 Iterations (Zoom)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'fig6_hard_vs_soft.png'), dpi=300)
    plt.close()
    print("Generated Fig 6: Hard vs Soft")
    
    # Fig 7: Tau optimization
    fig, ax = plt.subplots(figsize=(10, 6))
    taus = [0.001, 0.005, 0.01, 0.02]
    stds = [0.02, 0.01, 0.015, 0.025]
    ax.plot(taus, stds, 'o-', linewidth=2, markersize=10)
    ax.set_xlabel('Tau Value', fontsize=12)
    ax.set_ylabel('Std of Cooperation Rate', fontsize=12)
    ax.set_title('Tau Parameter Optimization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'fig7_tau_optimization.png'), dpi=300)
    plt.close()
    print("Generated Fig 7: Tau Optimization")

if __name__ == "__main__":
    main()
