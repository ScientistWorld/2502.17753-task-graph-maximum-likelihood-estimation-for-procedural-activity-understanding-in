#!/bin/bash
# Train-slice evaluation entry point.
# Auto-discovers all methods in checkpoint directories and scores them.
#
# Usage:
#   ./evaluate_train.sh                    # auto-discover all methods
#   ./evaluate_train.sh checkpoints/do     # evaluate specific method

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

CHECKPOINT_DIR="${1:-}"  # If empty, auto-discover

# Auto-discover all method directories with prediction files
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Auto-discovering methods in checkpoints/..."
    METHODS=()

    # Find all directories containing _predictions.npz files
    for dir in /home/user/checkpoints/*/; do
        if [ -d "$dir" ] && ls "$dir"/*_predictions.npz 1>/dev/null 2>&2; then
            dirname=$(basename "$dir")
            METHODS+=("$dirname")
        fi
    done

    if [ ${#METHODS[@]} -eq 0 ]; then
        echo "No methods found with prediction files."
        exit 0
    fi

    echo "Found methods: ${METHODS[*]}"

    # Always run count_based first (no checkpoint needed)
    echo "Evaluating count_based..."
    python3 /home/user/eval/train/evaluate.py captaincook4d_count checkpoints/ 5 || true

    # Then evaluate each discovered DO method
    for method in "${METHODS[@]}"; do
        echo "Evaluating $method..."
        python3 /home/user/eval/train/evaluate.py "$method" "checkpoints/$method" 5 || true
    done
else
    # Specific method provided
    if [[ "$CHECKPOINT_DIR" == *"count"* ]]; then
        echo "Evaluating count_based..."
        python3 /home/user/eval/train/evaluate.py captaincook4d_count checkpoints/ 5
    else
        echo "Evaluating $CHECKPOINT_DIR..."
        python3 /home/user/eval/train/evaluate.py "$CHECKPOINT_DIR" "checkpoints/$CHECKPOINT_DIR" 5
    fi
fi

# Add slice marker to scores_train.json
python3 - <<'PY'
import json
import os

scores_path = "scoring/scores_train.json"
if os.path.exists(scores_path):
    with open(scores_path) as f:
        scores = json.load(f)
    scores["slice"] = "train"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")
    print(f"Added slice='train' marker to {scores_path}")
PY

echo "Train-slice evaluation complete."
