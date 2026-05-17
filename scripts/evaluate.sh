#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script is self-contained using only eval/ and data/. It reads prediction
# artifacts written by a method and writes scoring/scores.json in the structure
# expected by scoring/reference.json.
#
# Usage:
#   ./evaluate.sh                                      # score all discoverable methods
#   ./evaluate.sh count_based                         # score built-in baseline
#   ./evaluate.sh do checkpoints/do_captaincook4d_fixedtgml
#   ./evaluate.sh my_method checkpoints/my_method
#   ./evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5

set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

mkdir -p scoring

# Standalone evaluation should work both inside the clean compute container
# and on the login node, where only the mounted user-site NumPy/NetworkX stack
# may be available. Keep run.sh strict; only evaluation gets this fallback.
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
METHOD_LIST="/tmp/evaluate_methods.tsv"

prediction_count() {
    local dir="$1"
    find "$dir" -maxdepth 1 -name '*_predictions.npz' 2>/dev/null | wc -l
}

canonical_method_name() {
    local method_name="$1"
    method_name="$(basename "$method_name")"
    case "$method_name" in
        count|count_based|captaincook4d_count|captaincook4d_count_based)
            echo "count_based"
            ;;
        do|captaincook4d_do|do_captaincook4d_fixedtgml)
            echo "do"
            ;;
        captaincook4d_*)
            echo "${method_name#captaincook4d_}"
            ;;
        *)
            echo "$method_name"
            ;;
    esac
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

    if [ "$method_name" = "do" ]; then
        candidate="checkpoints/do_captaincook4d_fixedtgml"
        if [ "$(prediction_count "$candidate")" -eq "$EXPECTED_PREDICTIONS" ]; then
            echo "$candidate"
            return 0
        fi
    fi

    return 1
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

if not results:
    raise SystemExit(1)

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

run_eval() {
    local method_name="$1"
    local checkpoint_dir="$2"
    local min_seeds="$3"
    local row_name
    row_name="$(canonical_method_name "$method_name")"

    echo "Evaluating ${row_name}..."
    python3 - "$row_name" "$checkpoint_dir" "$min_seeds" <<'PY'
import sys

sys.path.insert(0, "/home/user")

from data.loader import load_data
from eval.evaluator import evaluate_all_scenarios, write_scores_json

method_name = sys.argv[1]
checkpoint_dir = sys.argv[2]
min_seeds = int(sys.argv[3])

repo_path = "/home/user/data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d"
data = load_data(repo_data_path=repo_path)
print(f"Loaded {len(data)} scenarios")

if method_name == "count_based":
    results = evaluate_all_scenarios(data, method="count_based")
else:
    if not checkpoint_dir:
        raise SystemExit(f"{method_name} evaluation requires a prediction directory.")
    results = evaluate_all_scenarios(
        data,
        do_predictions_dir=checkpoint_dir,
        method="do",
        min_do_seeds=min_seeds,
    )

print(f"{method_name}: F1={results['aggregate']['mean_f1']:.1f}")
write_scores_json(
    results,
    "captaincook4d_task_graph",
    dataset_override="captaincook4d_task_graph",
    method_override=method_name,
)
PY
}

if [ "$METHOD_NAME" = "captaincook4d_partial_compare" ]; then
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "Partial comparison requires a checkpoint/results directory." >&2
        exit 1
    fi
    python3 - "$CHECKPOINT_DIR" "$MIN_SEEDS" <<'PY'
import sys

sys.path.insert(0, "/home/user")

from data.loader import load_data
from eval.evaluator import evaluate_all_scenarios, write_scores_json

checkpoint_dir = sys.argv[1]
min_seeds = int(sys.argv[2])

repo_path = "/home/user/data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d"
data = load_data(repo_data_path=repo_path)

do_results = evaluate_all_scenarios(
    data,
    do_predictions_dir=checkpoint_dir,
    method="do",
    allow_partial=True,
    min_do_seeds=min_seeds,
)
completed = set(do_results["per_scenario"].keys())
if not completed:
    raise SystemExit("No scenarios with sufficient corrected DO seeds are available yet.")

subset_data = [scenario for scenario in data if scenario.scenario_name in completed]
count_results = evaluate_all_scenarios(subset_data, method="count_based")

print(f"Completed subset: {len(completed)} scenarios -> {sorted(completed)}")
print(f"Partial Count-Based: F1={count_results['aggregate']['mean_f1']:.1f}")
print(f"Partial DO: F1={do_results['aggregate']['mean_f1']:.1f}")

write_scores_json(
    count_results,
    "captaincook4d_count_partial",
    dataset_override="captaincook4d_task_graph",
    method_override="count_based",
)
write_scores_json(
    do_results,
    "captaincook4d_do_partial",
    dataset_override="captaincook4d_task_graph",
    method_override="do",
)
PY
elif [ -z "$METHOD_NAME" ]; then
    rm -f "$METHOD_LIST"
    if methods_from_scores_json > "$METHOD_LIST"; then
        echo "Scoring methods from scoring/scores.json..."
    else
        echo "scoring/scores.json not found; falling back to checkpoint discovery."
        printf "count_based\t\n" > "$METHOD_LIST"
        while IFS= read -r discovered; do
            [ -n "$discovered" ] || continue
            printf "%s\tcheckpoints/%s\n" "$discovered" "$discovered" >> "$METHOD_LIST"
        done < <(discover_prediction_methods)
    fi

    rm -f scoring/scores.json
    while IFS=$'\t' read -r score_method score_checkpoint; do
        [ -n "$score_method" ] || continue
        score_method="$(canonical_method_name "$score_method")"
        if [ "$score_method" = "count_based" ]; then
            run_eval "$score_method" "" "$MIN_SEEDS"
        else
            checkpoint_dir="$(resolve_checkpoint_dir "$score_method" "$score_checkpoint")" || {
                echo "No complete prediction directory found for ${score_method}" >&2
                exit 1
            }
            run_eval "$score_method" "$checkpoint_dir" "$MIN_SEEDS"
        fi
    done < "$METHOD_LIST"
else
    METHOD_NAME="$(canonical_method_name "$METHOD_NAME")"
    if [ "$METHOD_NAME" = "count_based" ]; then
        run_eval "$METHOD_NAME" "" "$MIN_SEEDS"
    else
        if [ -z "$CHECKPOINT_DIR" ] && [ -d "$1" ]; then
            CHECKPOINT_DIR="$1"
        fi
        CHECKPOINT_DIR="$(resolve_checkpoint_dir "$METHOD_NAME" "$CHECKPOINT_DIR")" || {
            echo "No complete prediction directory found for ${METHOD_NAME}" >&2
            exit 1
        }
        run_eval "$METHOD_NAME" "$CHECKPOINT_DIR" "$MIN_SEEDS"
    fi
fi

echo "Evaluation complete. Scores written to scoring/scores.json."
