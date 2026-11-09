#!/bin/bash
# Train (public) slice: scores every method directory under checkpoints/ (any
# directory holding <procedure>_predictions.npz files) plus the built-in
# Count-Based baseline on the 12 public CaptainCook4D procedures, and writes
# scoring/scores_train.json from scratch. Same contract and metrics as the
# hidden test. No arguments needed.
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

mkdir -p scoring
rm -f scoring/scores_train.json
python3 /home/user/eval/train/evaluate.py "$@"
echo "Train-slice evaluation complete."
