#!/bin/bash
# Continue the corrected CaptainCook4D TGML/DO rerun and refresh scores.
set -e
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === Continue corrected DO rerun ==="

# Bank progress conservatively: repeated cluster failures have killed longer
# jobs, so each submission completes at most two pending procedures before
# refreshing the like-for-like partial comparison.
export METHOD_DEVICE=cpu
export METHOD_MAX_EPOCHS="${METHOD_MAX_EPOCHS:-1000}"
export METHOD_SEEDS="${METHOD_SEEDS:-42,1337,2024,2025,2026}"
export METHOD_MAX_SCENARIOS="${METHOD_MAX_SCENARIOS:-2}"
export METHOD_OUTPUT_DIR="${METHOD_OUTPUT_DIR:-checkpoints/do_captaincook4d_fixedtgml}"

echo "Training pending DO scenarios..."
bash scripts/method.sh

echo "Refreshing corrected partial comparison..."
bash scripts/evaluate.sh captaincook4d_partial_compare "$METHOD_OUTPUT_DIR" 5

python3 validate.py --compare

echo "=== Done ==="
