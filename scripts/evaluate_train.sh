#!/bin/bash
# Train-slice evaluation entry point.
# Auto-discovers all methods in checkpoint directories and scores them.
#
# Usage:
#   ./evaluate_train.sh                    # auto-discover all complete methods
#   ./evaluate_train.sh do_captaincook4d   # evaluate a specific method
#   ./evaluate_train.sh checkpoints/do_captaincook4d

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
EXPECTED_PREDICTIONS=24

# Auto-discover all complete method directories with prediction files.
# Require full coverage so incomplete smoke runs do not silently mutate scores.
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Auto-discovering methods in checkpoints/..."
    METHODS=()

    for dir in /home/user/checkpoints/*/; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -maxdepth 1 -name '*_predictions.npz' | wc -l)
            if [ "$count" -eq "$EXPECTED_PREDICTIONS" ]; then
                METHODS+=("$(basename "$dir")")
            elif [ "$count" -gt 0 ]; then
                echo "Skipping $(basename "$dir") ($count/$EXPECTED_PREDICTIONS prediction files)"
            fi
        fi
    done

    if [ ${#METHODS[@]} -eq 0 ]; then
        echo "No methods found with prediction files."
        exit 0
    fi

    echo "Found methods: ${METHODS[*]}"

    # Always run count_based first (no checkpoint needed)
    echo "Evaluating count_based..."
    python3 /home/user/eval/train/evaluate.py captaincook4d_count checkpoints/ 5

    # Then evaluate each discovered DO method
    for method in "${METHODS[@]}"; do
        echo "Evaluating $method..."
        python3 /home/user/eval/train/evaluate.py "$method" "checkpoints/$method" 5
    done
else
    # Specific method provided
    if [[ "$CHECKPOINT_DIR" == *"count"* ]]; then
        echo "Evaluating count_based..."
        python3 /home/user/eval/train/evaluate.py captaincook4d_count checkpoints/ 5
    else
        if [ -d "$CHECKPOINT_DIR" ]; then
            METHOD_NAME="$(basename "$CHECKPOINT_DIR")"
            PRED_DIR="$CHECKPOINT_DIR"
        else
            METHOD_NAME="$(basename "$CHECKPOINT_DIR")"
            PRED_DIR="checkpoints/$CHECKPOINT_DIR"
        fi
        echo "Evaluating $METHOD_NAME..."
        python3 /home/user/eval/train/evaluate.py "$METHOD_NAME" "$PRED_DIR" 5
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
