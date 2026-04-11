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

# Step 2: DO method (3 seeds, 300 epochs each)
echo "[2/2] Running DO method (seeds 42, 123, 456)..."
bash scripts/method.sh

echo ""
echo "=== Reproduction complete ==="
echo "Results in scoring/scores.json"
python3 validate.py --compare
