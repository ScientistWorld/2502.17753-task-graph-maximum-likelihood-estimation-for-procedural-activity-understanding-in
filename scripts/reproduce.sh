#!/bin/bash
# Reproduce the reference end to end (CPU only). Run scripts/download.sh first
# (compute nodes have no internet).
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== Reproducing the TGML Direct Optimization reference ==="
rm -rf checkpoints/do
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate.sh
python3 validate.py
echo "=== Reproduction complete: scoring/scores.json, scoring/scores_train.json ==="
