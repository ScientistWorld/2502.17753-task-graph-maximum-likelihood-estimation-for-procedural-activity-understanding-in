#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh's contract:
#   ./evaluate_test.sh METHOD_NAME CHECKPOINT_DIR MIN_SEEDS
#
# With no arguments, reads scoring/scores_train.json and scores every method key
# present there on the held-out eval/test/ slice.

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

run_test_eval() {
    local method_name="$1"
    local checkpoint_dir="$2"
    local min_seeds="$3"

    echo "Evaluating ${method_name} on test slice..."
    python3 /home/user/eval/test/evaluate.py "$method_name" "$checkpoint_dir" "$min_seeds"
}

methods_from_train_scores() {
    python3 - <<'PY'
import json
import os

path = "scoring/scores_train.json"
if not os.path.exists(path):
    raise SystemExit(1)

with open(path) as f:
    scores = json.load(f)

experiment = scores.get("experiments", {}).get("captaincook4d_task_graph", {})
results = experiment.get("results", {})
artifacts = scores.get("method_artifacts", {})

for method_name in results:
    checkpoint_dir = artifacts.get(method_name, {}).get("checkpoint_dir", "")
    print(f"{method_name}\t{checkpoint_dir}")
PY
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
    rm -f scoring/scores_test.json
    if methods_from_train_scores >/tmp/evaluate_test_methods.tsv; then
        while IFS=$'\t' read -r train_method train_checkpoint; do
            [ -n "$train_method" ] || continue
            if [[ "$train_method" == *"count"* ]]; then
                run_test_eval "$train_method" "" "$MIN_SEEDS"
            else
                if [ -z "$train_checkpoint" ]; then
                    train_checkpoint="checkpoints/$train_method"
                fi
                run_test_eval "$train_method" "$train_checkpoint" "$MIN_SEEDS"
            fi
        done < /tmp/evaluate_test_methods.tsv
    else
        echo "scoring/scores_train.json not found; falling back to checkpoint discovery."
        mapfile -t METHODS < <(discover_prediction_methods)

        run_test_eval "count_based" "" "$MIN_SEEDS"

        if [ ${#METHODS[@]} -eq 0 ]; then
            echo "No complete prediction methods found in checkpoints/."
            exit 0
        fi

        for method in "${METHODS[@]}"; do
            run_test_eval "$method" "checkpoints/$method" "$MIN_SEEDS"
        done
    fi
else
    if [[ "$METHOD_NAME" == *"count"* ]]; then
        run_test_eval "$METHOD_NAME" "" "$MIN_SEEDS"
    else
        if [ -z "$CHECKPOINT_DIR" ]; then
            if [ -d "$METHOD_NAME" ]; then
                CHECKPOINT_DIR="$METHOD_NAME"
                METHOD_NAME="$(basename "$METHOD_NAME")"
            else
                CHECKPOINT_DIR="checkpoints/$METHOD_NAME"
            fi
        fi
        run_test_eval "$METHOD_NAME" "$CHECKPOINT_DIR" "$MIN_SEEDS"
    fi
fi

echo "Test-slice evaluation complete."
