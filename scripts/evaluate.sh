#!/bin/bash
# Development evaluation (public ground truth only): scores every method
# directory under checkpoints/ plus the built-in Count-Based baseline on the 12
# CaptainCook4D procedures whose graphs are public, and writes scoring/scores.json.
# The 12 held-out procedures are scored only by the hidden test.
#
# Prediction contract (see briefing/evaluation.md): for each procedure,
#   checkpoints/<method>/<procedure>_predictions.npz  with  pred_cores: (5, n, n)
# finite weighted adjacency values in [0, 1].
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

mkdir -p scoring
python3 /home/user/eval/train/evaluate.py --output /home/user/scoring/scores.json "$@"
