#!/bin/bash
set -e

echo "=== Waiting for coupling experiment (PID 4312) to finish ==="
while kill -0 4312 2>/dev/null; do
    sleep 10
done
echo "Coupling experiment done at $(date)"

# Check if coupling results were saved
if [ -f ~/pavo/experiments/outputs/coupling_results_200.json ]; then
    echo "Coupling results saved successfully"
else
    echo "WARNING: Coupling results not found"
fi

echo ""
echo "=== Running Experiment 4: Real Ablation ==="
cd ~/pavo/experiments
python3 exp4_real_ablation.py 2>&1

echo ""
echo "=== All experiments complete at $(date) ==="
echo "Output files:"
ls -la ~/pavo/experiments/outputs/
