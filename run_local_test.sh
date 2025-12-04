#!/bin/bash
# Quick local test to generate sample figures
# Designed for 6-core/12-thread machine
# Expected runtime: ~10-15 minutes

set -e
cd "$(dirname "$0")"

echo "========================================"
echo "Quick Local Test for Paper Figures"
echo "========================================"
echo ""

# Create output directories
mkdir -p local_test/exp1_core
mkdir -p local_test/exp2_baseline
mkdir -p local_test/paper_figures

# Short iterations for quick test
ITERATIONS=100000
L=20  # Smaller grid for speed

echo "Settings: L=$L, iterations=$ITERATIONS"
echo ""

# ============================================
# EXP1: Core dual-brain (parallel with 6 jobs)
# ============================================
echo "[1/3] Running core experiments (λ sweep)..."

run_exp1() {
    local lam=$1
    local r=$2
    echo "  Running λ=$lam, r=$r..."
    python3 run_experiment.py \
        --r $r \
        --dqn_lambda $lam \
        --iterations $ITERATIONS \
        --L $L \
        --influence_factor 0.0 \
        --use_soft_update \
        --dqn_tau 0.005 \
        --output_dir local_test/exp1_core \
        2>/dev/null
}

# Run in parallel (6 jobs at a time)
for lam in 0.0 0.5 1.0; do
    for r in 2.6 4.0; do
        run_exp1 $lam $r &
    done
    wait
done

echo "  Core experiments done!"

# ============================================
# EXP2: Baseline comparison
# ============================================
echo ""
echo "[2/3] Running baseline experiments..."

run_baseline() {
    local method=$1
    local r=$2
    echo "  Running $method, r=$r..."
    python3 experiments/run_baseline.py \
        --method $method \
        --r $r \
        --iterations $ITERATIONS \
        --L $L \
        --output_dir local_test/exp2_baseline \
        2>/dev/null
}

# Run baselines in parallel
for r in 4.0; do
    run_baseline q_only $r &
    run_baseline dqn_only $r &
    run_baseline fermi $r &
    run_baseline imitation $r &
    run_baseline dual_brain $r &
done
wait

echo "  Baseline experiments done!"

# ============================================
# Generate Plots
# ============================================
echo ""
echo "[3/3] Generating figures..."

export PYTHONPATH=$PYTHONPATH:.

# Core plots
echo "  Generating core plots..."
python3 src/visualization/plot_core.py \
    --input local_test/exp1_core/* \
    --output local_test/paper_figures/ 2>/dev/null || echo "  (plot_core had warnings)"

# Baseline plots
echo "  Generating baseline plots..."
python src/visualization/plot_baseline.py \
    --input local_test/exp2_baseline/* \
    --output local_test/paper_figures/ 2>/dev/null || echo "  (plot_baseline had warnings)"

# Q-value plots
echo "  Generating Q-value plots..."
python src/visualization/plot_qvalue.py \
    --input local_test/exp1_core/* \
    --output local_test/paper_figures/ 2>/dev/null || echo "  (plot_qvalue had warnings)"

echo ""
echo "========================================"
echo "Done! Figures saved to: local_test/paper_figures/"
echo "========================================"
ls -la local_test/paper_figures/
