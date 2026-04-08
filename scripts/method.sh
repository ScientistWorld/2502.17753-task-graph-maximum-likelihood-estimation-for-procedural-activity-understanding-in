#!/bin/bash
# Run the paper's proposed DO method.
#
# Usage:
#   bash scripts/method.sh

set -e
cd /home/user

echo "Running Direct Optimization (DO) model on CaptainCook4D..."

METHOD_DEVICE="${METHOD_DEVICE:-cpu}"
METHOD_MAX_EPOCHS="${METHOD_MAX_EPOCHS:-1000}"
METHOD_SEEDS="${METHOD_SEEDS:-42}"

# Remove previous outputs so evaluation only sees results from this run.
rm -rf checkpoints/do_captaincook4d

# Train DO on CaptainCook4D
python3 method/train.py \
    --dataset captaincook4d \
    --repo_data_path /home/user/method/repo/data/captaincook4d \
    --output checkpoints/do_captaincook4d \
    --beta 0.005 \
    --lr 0.1 \
    --max_epochs "${METHOD_MAX_EPOCHS}" \
    --seeds ${METHOD_SEEDS} \
    --device "${METHOD_DEVICE}" \
    --verbose

echo "Training complete. Evaluating..."
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d/

echo "Done."
