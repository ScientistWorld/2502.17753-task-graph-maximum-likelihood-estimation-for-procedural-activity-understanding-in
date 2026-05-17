#!/bin/bash
# Train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh's contract:
#   ./evaluate_train.sh METHOD_NAME CHECKPOINT_DIR MIN_SEEDS
#
# With no arguments, scores every method key already present in scores.json.
# The score file records the checkpoint path for each method so
# scripts/evaluate_test.sh can score the exact same method keys later.

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

prediction_count() {
    local dir="$1"
    find "$dir" -maxdepth 1 -name '*_predictions.npz' 2>/dev/null | wc -l
}

methods_from_scores_json() {
    python3 - <<'PY'
import json
import os

path = "scoring/scores.json"
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

resolve_checkpoint_dir() {
    local method_name="$1"
    local checkpoint_dir="$2"
    local candidate

    if [ -n "$checkpoint_dir" ]; then
        echo "$checkpoint_dir"
        return 0
    fi

    candidate="checkpoints/$method_name"
    if [ "$(prediction_count "$candidate")" -eq "$EXPECTED_PREDICTIONS" ]; then
        echo "$candidate"
        return 0
    fi

    # The canonical paper-method row is named "do" in scores.json, while the
    # full corrected artifact directory is versioned to preserve rerun history.
    if [ "$method_name" = "do" ]; then
        candidate="checkpoints/do_captaincook4d_fixedtgml"
        if [ "$(prediction_count "$candidate")" -eq "$EXPECTED_PREDICTIONS" ]; then
            echo "$candidate"
            return 0
        fi
    fi

    return 1
}

canonical_method_name() {
    local method_name="$1"
    if [ "$method_name" = "do_captaincook4d_fixedtgml" ]; then
        echo "do"
    else
        echo "$method_name"
    fi
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
    if methods_from_scores_json >/tmp/evaluate_train_methods.tsv; then
        echo "Scoring train-slice methods from scoring/scores.json..."
        while IFS=$'\t' read -r score_method score_checkpoint; do
            [ -n "$score_method" ] || continue
            if [[ "$score_method" == *"count"* ]]; then
                run_train_eval "count_based" "" "$MIN_SEEDS"
            else
                checkpoint_dir="$(resolve_checkpoint_dir "$score_method" "$score_checkpoint")" || {
                    echo "No complete prediction directory found for ${score_method}" >&2
                    exit 1
                }
                run_train_eval "$score_method" "$checkpoint_dir" "$MIN_SEEDS"
            fi
        done < /tmp/evaluate_train_methods.tsv
    else
        echo "scoring/scores.json not found; falling back to checkpoint discovery."
        mapfile -t METHODS < <(discover_prediction_methods)

        run_train_eval "count_based" "" "$MIN_SEEDS"

        if [ ${#METHODS[@]} -eq 0 ]; then
            echo "No complete prediction methods found in checkpoints/."
            exit 0
        fi

        for method in "${METHODS[@]}"; do
            run_train_eval "$method" "checkpoints/$method" "$MIN_SEEDS"
        done
    fi
else
    if [[ "$METHOD_NAME" == *"count"* ]]; then
        run_train_eval "count_based" "" "$MIN_SEEDS"
    else
        if [ -z "$CHECKPOINT_DIR" ]; then
            if [ -d "$METHOD_NAME" ]; then
                CHECKPOINT_DIR="$METHOD_NAME"
                METHOD_NAME="$(canonical_method_name "$(basename "$METHOD_NAME")")"
            else
                CHECKPOINT_DIR="checkpoints/$METHOD_NAME"
            fi
        fi
        run_train_eval "$METHOD_NAME" "$CHECKPOINT_DIR" "$MIN_SEEDS"
    fi
fi

echo "Train-slice evaluation complete."
