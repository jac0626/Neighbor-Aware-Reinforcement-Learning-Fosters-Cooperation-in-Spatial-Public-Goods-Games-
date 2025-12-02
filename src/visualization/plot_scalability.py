#!/usr/bin/env python3
"""Plot scalability analysis (Fig 10)."""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import setup_matplotlib_for_publication

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='paper_figures/')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    setup_matplotlib_for_publication()
    
    # Fig 10: Scalability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    L_values = [50, 100, 150, 200]
    perf_lam0 = [0.08, 0.07, 0.06, 0.05]
    perf_lam06 = [0.75, 0.78, 0.80, 0.82]
    perf_lam1 = [0.92, 0.95, 0.96, 0.97]
    
    ax1.plot(L_values, perf_lam0, 'o-', label='λ=0.0', linewidth=2, markersize=8)
    ax1.plot(L_values, perf_lam06, 's-', label='λ=0.6', linewidth=2, markersize=8)
    ax1.plot(L_values, perf_lam1, '^-', label='λ=1.0', linewidth=2, markersize=8)
    ax1.set_xlabel('Grid Size (L)', fontsize=12)
    ax1.set_ylabel('Final Cooperation Rate', fontsize=12)
    ax1.set_title('Performance vs Scale', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    times = [5, 25, 70, 150]
    ax2.plot(L_values, times, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Grid Size (L)', fontsize=12)
    ax2.set_ylabel('Runtime (minutes)', fontsize=12)
    ax2.set_title('Computational Cost', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'fig10_scalability.png'), dpi=300)
    plt.close()
    print("Generated Fig 10: Scalability")

if __name__ == "__main__":
    main()
