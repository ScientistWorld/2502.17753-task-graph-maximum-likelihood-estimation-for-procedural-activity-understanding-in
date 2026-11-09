#!/bin/bash
# Hidden test: scores every method directory under checkpoints/ (any directory
# holding <procedure>_predictions.npz files) plus the evaluator's own Count-Based
# baseline on the 12 held-out CaptainCook4D procedures, and writes
# scoring/scores_test.json from scratch. No arguments needed; no method names,
# registries or score files written by a method are read.
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

mkdir -p scoring
rm -f scoring/scores_test.json
python3 /home/user/eval/test/evaluate.py "$@"
echo "Test-slice evaluation complete."
