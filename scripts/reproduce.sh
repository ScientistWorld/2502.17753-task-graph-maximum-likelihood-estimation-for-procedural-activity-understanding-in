#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user

set -e
cd /home/user

echo "=== Reproducing TGML Paper Results ==="
echo ""

# Start from a clean score file so this run cannot inherit stale numbers.
rm -f scoring/scores.json

# Step 1: Run Count-Based baseline
echo "[1/2] Running Count-Based baseline..."
bash scripts/baseline.sh

# Step 2: Run DO method
echo "[2/2] Running DO method..."
bash scripts/method.sh

echo ""
echo "=== Reproduction complete ==="
echo "Results in scoring/scores.json"
