#!/bin/bash
# Run the paper's proposed DO method.
#
# Usage:
#   bash scripts/method.sh

set -e
cd /home/user

echo "Running Direct Optimization (DO) model on CaptainCook4D..."

# Remove previous outputs so evaluation only sees results from this run.
rm -rf checkpoints/do_captaincook4d

# Train DO on CaptainCook4D
python3 method/train.py \
    --dataset captaincook4d \
    --repo_data_path /home/user/method/repo/data/captaincook4d \
    --output checkpoints/do_captaincook4d \
    --beta 0.005 \
    --lr 0.1 \
    --max_epochs 1000 \
    --seeds 42 \
    --device cuda \
    --verbose

echo "Training complete. Evaluating..."
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d/

echo "Done."
