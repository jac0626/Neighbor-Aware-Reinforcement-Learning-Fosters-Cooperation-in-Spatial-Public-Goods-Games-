#!/bin/bash
# Quick local test for phase diagram generation
# Uses small grid (10x10) with few iterations for speed

set -e
cd "$(dirname "$0")"

echo "========================================"
echo "Phase Diagram Local Test"
echo "========================================"

# Settings for quick test
LAMBDA_POINTS=10
R_POINTS=10
CHUNK_SIZE=5
ITERATIONS=500
L=20

rm -rf phase_test_results phase_test_figures
mkdir -p phase_test_results phase_test_figures

echo "Settings: ${LAMBDA_POINTS}x${R_POINTS} grid, ${ITERATIONS} iterations, L=${L}"
echo ""

# Run 4 chunks (2x2) to simulate the workflow
echo "[1/3] Running experiment chunks..."

for lam_chunk in 0 1; do
    for r_chunk in 0 1; do
        echo "  Chunk (λ=$lam_chunk, r=$r_chunk)..."
        python3 experiments/run_phase_diagram.py \
            --lambda_chunk $lam_chunk \
            --r_chunk $r_chunk \
            --lambda_points $LAMBDA_POINTS \
            --r_points $R_POINTS \
            --chunk_size $CHUNK_SIZE \
            --iterations $ITERATIONS \
            --L $L \
            --output_dir phase_test_results 2>/dev/null &
    done
done
wait

echo "  All chunks done!"
echo ""

# Generate plots
echo "[2/3] Generating phase diagram..."
export PYTHONPATH=$PYTHONPATH:.

python3 src/visualization/plot_phase_diagram.py \
    --input phase_test_results/ \
    --output phase_test_figures/ \
    --lambda_points $LAMBDA_POINTS \
    --r_points $R_POINTS

echo ""
echo "[3/3] Results"
echo "========================================"
ls -la phase_test_figures/
echo ""
echo "Test complete!"
