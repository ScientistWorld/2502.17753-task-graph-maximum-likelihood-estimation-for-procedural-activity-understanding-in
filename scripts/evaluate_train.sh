#!/bin/bash
# Train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh's contract:
#   ./evaluate_train.sh METHOD_NAME CHECKPOINT_DIR MIN_SEEDS
#
# With no arguments, scores count_based plus every complete prediction directory
# under checkpoints/. The score file records the checkpoint path for each method
# so scripts/evaluate_test.sh can score the exact same method keys later.

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

METHOD_NAME="${1:-}"
CHECKPOINT_DIR="${2:-}"
MIN_SEEDS="${3:-5}"
EXPECTED_PREDICTIONS=24

run_train_eval() {
    local method_name="$1"
    local checkpoint_dir="$2"
    local min_seeds="$3"

    echo "Evaluating ${method_name} on train slice..."
    python3 /home/user/eval/train/evaluate.py "$method_name" "$checkpoint_dir" "$min_seeds"
}

discover_prediction_methods() {
    local methods=()
    local dir count

    for dir in /home/user/checkpoints/*/; do
        [ -d "$dir" ] || continue
        count=$(find "$dir" -maxdepth 1 -name '*_predictions.npz' | wc -l)
        if [ "$count" -eq "$EXPECTED_PREDICTIONS" ]; then
            methods+=("$(basename "$dir")")
        elif [ "$count" -gt 0 ]; then
            echo "Skipping $(basename "$dir") ($count/$EXPECTED_PREDICTIONS prediction files)" >&2
        fi
    done

    printf '%s\n' "${methods[@]}"
}

if [ -z "$METHOD_NAME" ]; then
    rm -f scoring/scores_train.json
    echo "Auto-discovering train-slice methods in checkpoints/..."
    mapfile -t METHODS < <(discover_prediction_methods)

    run_train_eval "count_based" "" "$MIN_SEEDS"

    if [ ${#METHODS[@]} -eq 0 ]; then
        echo "No complete prediction methods found in checkpoints/."
        exit 0
    fi

    for method in "${METHODS[@]}"; do
        run_train_eval "$method" "checkpoints/$method" "$MIN_SEEDS"
    done
else
    if [[ "$METHOD_NAME" == *"count"* ]]; then
        run_train_eval "count_based" "" "$MIN_SEEDS"
    else
        if [ -z "$CHECKPOINT_DIR" ]; then
            if [ -d "$METHOD_NAME" ]; then
                CHECKPOINT_DIR="$METHOD_NAME"
                METHOD_NAME="$(basename "$METHOD_NAME")"
            else
                CHECKPOINT_DIR="checkpoints/$METHOD_NAME"
            fi
        fi
        run_train_eval "$METHOD_NAME" "$CHECKPOINT_DIR" "$MIN_SEEDS"
    fi
fi

echo "Train-slice evaluation complete."
