#!/bin/bash
# Train the paper's Direct Optimization (DO) model: TGML loss, Adam lr 0.1,
# beta 0.005, at most 1000 epochs, Sequence-Accuracy early stopping (patience 25),
# five seeds per procedure, all 24 CaptainCook4D procedures (CPU, a few minutes).
# Writes checkpoints/do/<procedure>_predictions.npz.
#
# Usage:
#   bash scripts/method.sh
#   METHOD_OUTPUT_DIR=checkpoints/do_beta05 bash scripts/method.sh --beta 0.5
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

OUTPUT_DIR="${METHOD_OUTPUT_DIR:-checkpoints/do}"
python3 method/train.py --output "$OUTPUT_DIR" "$@"
