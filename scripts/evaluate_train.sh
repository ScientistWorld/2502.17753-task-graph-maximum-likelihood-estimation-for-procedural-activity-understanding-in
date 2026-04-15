#!/bin/bash
# Train-slice evaluation entry point. Mirrors scripts/evaluate.sh argument shape:
#   ./evaluate_train.sh captaincook4d_do checkpoints/do_captaincook4d
#   ./evaluate_train.sh captaincook4d_count

set -e
cd /home/user
source /home/user/scripts/common_env.sh

mkdir -p scoring

if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
import networkx
PY
then
  export PYTHONPATH="${PYTHONPATH}:/home/user/.local/lib/python3.12/site-packages"
fi

METHOD_NAME="${1:-captaincook4d_do}"
CHECKPOINT_DIR="${2:-checkpoints/}"
MIN_SEEDS="${3:-5}"

python3 /home/user/eval/train/evaluate.py "$METHOD_NAME" "$CHECKPOINT_DIR" "$MIN_SEEDS"

echo "Train-slice evaluation complete."
