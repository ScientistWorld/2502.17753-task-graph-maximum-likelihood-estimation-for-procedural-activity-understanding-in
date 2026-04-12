#!/bin/bash
# Reproduce the paper's results end-to-end.
# Run this inside the compute container after running scripts/download.sh.

set -e
cd /home/user

echo "=== Reproducing TGML Paper Results ==="

# Clean slate
rm -f scoring/scores.json

# Step 1: Count-Based baseline
echo "[1/2] Running Count-Based baseline..."
bash scripts/baseline.sh

# Step 2: DO method
echo "[2/2] Running DO method (5 seeds, 1000 epochs each)..."
METHOD_DEVICE=cpu METHOD_SEEDS=42,1337,2024,2025,2026 METHOD_MAX_EPOCHS=1000 \
METHOD_OUTPUT_DIR=checkpoints/do_captaincook4d_fixedtgml \
bash scripts/method.sh

echo ""
echo "=== Reproduction complete ==="
echo "Results in scoring/scores.json"
python3 validate.py --compare
