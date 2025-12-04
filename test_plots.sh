#!/bin/bash
# Quick plot test with minimal iterations to verify functionality
set -e
cd "$(dirname "$0")"

echo "========================================" 
echo "Quick Plot Test (Minimal Iterations)"
echo "========================================" 
echo ""

# Very small settings for quick test
ITERATIONS=100
L=10

mkdir -p test_plots/exp1
mkdir -p test_plots/exp2
mkdir -p test_plots/figures

echo "Settings: L=$L, iterations=$ITERATIONS"
echo ""

# ============================================
# Quick experiment runs
# ============================================
echo "[1/3] Running minimal experiments..."

# Core experiments (just 2 configs)
echo "  Core: lambda=0.5, r=4.0..."
python3 run_experiment.py \
    --r 4.0 --dqn_lambda 0.5 --iterations $ITERATIONS --L $L \
    --influence_factor 0.0 --use_soft_update --dqn_tau 0.005 \
    --output_dir test_plots/exp1 2>/dev/null

echo "  Core: lambda=1.0, r=4.0..."
python3 run_experiment.py \
    --r 4.0 --dqn_lambda 1.0 --iterations $ITERATIONS --L $L \
    --influence_factor 0.0 --use_soft_update --dqn_tau 0.005 \
    --output_dir test_plots/exp1 2>/dev/null

# Baseline experiments
echo "  Baseline: fermi r=4.0..."
python3 experiments/run_baseline.py \
    --method fermi --r 4.0 --iterations $ITERATIONS --L $L \
    --output_dir test_plots/exp2 2>/dev/null

echo "  Baseline: imitation r=4.0..."
python3 experiments/run_baseline.py \
    --method imitation --r 4.0 --iterations $ITERATIONS --L $L \
    --output_dir test_plots/exp2 2>/dev/null

echo "  Baseline: dual_brain r=4.0..."
python3 experiments/run_baseline.py \
    --method dual_brain --r 4.0 --iterations $ITERATIONS --L $L \
    --output_dir test_plots/exp2 2>/dev/null

echo "  All experiments done!"

# ============================================ 
# Test all plotting scripts
# ============================================
echo ""
echo "[2/3] Testing all plotting scripts..."
export PYTHONPATH=$PYTHONPATH:.

echo ""
echo "--- plot_core.py ---"
python3 src/visualization/plot_core.py \
    --input test_plots/exp1/* \
    --output test_plots/figures/ && echo "OK" || echo "FAILED"

echo ""
echo "--- plot_baseline.py ---"
python3 src/visualization/plot_baseline.py \
    --input test_plots/exp2/* \
    --output test_plots/figures/ && echo "OK" || echo "FAILED"

echo ""
echo "--- plot_qvalue.py ---"
python3 src/visualization/plot_qvalue.py \
    --input test_plots/exp1/* \
    --output test_plots/figures/ && echo "OK" || echo "FAILED"

# ============================================
# Summary
# ============================================
echo ""
echo "[3/3] Summary"
echo "========================================" 
echo "Generated figures:"
ls -la test_plots/figures/ 2>/dev/null || echo "(no figures generated)"
echo ""
echo "Test complete!"
