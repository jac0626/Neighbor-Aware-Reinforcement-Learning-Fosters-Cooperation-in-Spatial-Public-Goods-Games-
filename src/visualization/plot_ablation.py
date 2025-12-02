#!/usr/bin/env python3
"""Plot ablation study results (Fig 4-5)."""
import argparse
import os
import sys
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True, help='Input directories with exp2 results')
    parser.add_argument('--output', default='paper_figures/')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    setup_matplotlib_for_publication()
    
    # Fig 4: Feature ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ['Full', 'No Rep', 'No Neighbor', 'Only Payoff']
    # Placeholder data - will be replaced with real data loading
    performance = [0.8, 0.6, 0.5, 0.3]
    ax.bar(features, performance, color='#1f77b4', alpha=0.7)
    ax.set_ylabel('Final Cooperation Rate', fontsize=12)
    ax.set_title('Feature Ablation Study', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'fig4_feature_ablation.png'), dpi=300)
    plt.close()
    print("Generated Fig 4: Feature Ablation")
    
    # Fig 5: Lambda sweep curve
    fig, ax = plt.subplots(figsize=(10, 6))
    lambdas = np.linspace(0, 1, 11)
    # Placeholder data
    performance = 0.1 + 0.8 * (1 - (lambdas - 0.6)**2)
    ax.plot(lambdas, performance, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('DQN Weight (λ)', fontsize=12)
    ax.set_ylabel('Final Cooperation Rate', fontsize=12)
    ax.set_title('Lambda Parameter Sweep', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'fig5_lambda_sweep.png'), dpi=300)
    plt.close()
    print("Generated Fig 5: Lambda Sweep")

if __name__ == "__main__":
    main()
