#!/bin/bash
# Run a fresh CaptainCook4D TGML Direct Optimization experiment using the
# paper's weakly supervised sequence-accuracy selection signal, then score it.
set -e
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === CaptainCook4D DO sequence-selection rerun ==="

export METHOD_OUTPUT_DIR="checkpoints/do_captaincook4d_seqselect"
export METHOD_SELECTION="sequence_accuracy"
export METHOD_SEEDS="42,1337,2024,2025,2026"
export METHOD_MAX_EPOCHS="1000"
export METHOD_DEVICE="cpu"

bash scripts/method.sh
bash scripts/evaluate.sh captaincook4d_do "$METHOD_OUTPUT_DIR"
bash scripts/evaluate_train.sh do "$METHOD_OUTPUT_DIR"
bash scripts/evaluate_test.sh do "$METHOD_OUTPUT_DIR"

python3 validate.py --compare --post-design

echo "=== Done ==="
