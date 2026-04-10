#!/bin/bash
# Run the paper's proposed DO method (parallel across scenarios).
#
# Usage:
#   bash scripts/method.sh

set -e
cd /home/user

echo "Running Direct Optimization (DO) model on CaptainCook4D..."

METHOD_DEVICE="${METHOD_DEVICE:-cpu}"
METHOD_MAX_EPOCHS="${METHOD_MAX_EPOCHS:-300}"
METHOD_SEEDS="${METHOD_SEEDS:-42}"
METHOD_WORKERS="${METHOD_WORKERS:-24}"

# Remove previous outputs so evaluation only sees results from this run.
rm -rf checkpoints/do_captaincook4d

# Train DO on ALL 24 scenarios in parallel.
python3 method/train_parallel.py \
    --repo_data_path /home/user/method/repo/data/captaincook4d \
    --output checkpoints/do_captaincook4d \
    --beta 0.005 \
    --lr 0.1 \
    --max_epochs "${METHOD_MAX_EPOCHS}" \
    --seed "${METHOD_SEEDS}" \
    --num_workers "${METHOD_WORKERS}"

echo "Training complete. Evaluating..."
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d/

echo "Done."
