#!/bin/bash
# Run train/test split scoring over existing checkpoint prediction artifacts.
set -e
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === Evaluate train/test splits ==="

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

python3 validate.py --compare --post-design

echo "=== Done ==="
