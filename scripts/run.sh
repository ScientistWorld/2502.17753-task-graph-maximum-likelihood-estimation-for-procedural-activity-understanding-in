#!/bin/bash
# Final CPU verification for the CaptainCook4D TGML reproduction.
# This refreshes scores from saved artifacts and runs the structural validator.
set -e
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === CaptainCook4D final verification ==="

bash scripts/evaluate.sh captaincook4d_count
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d_fixedtgml
bash scripts/evaluate_train.sh do checkpoints/do_captaincook4d_fixedtgml
bash scripts/evaluate_test.sh do checkpoints/do_captaincook4d_fixedtgml

python3 validate.py --compare --post-design

echo "=== Done ==="
