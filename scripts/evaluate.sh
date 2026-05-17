#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, matching scoring/reference.json structure.
#
# Usage:
#   ./evaluate.sh captaincook4d_do checkpoints/do_captaincook4d_fixedtgml
#   ./evaluate.sh captaincook4d_count     # evaluate count-based on CaptainCook4D
#   ./evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5

set -e
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

# Parse arguments
METHOD_NAME="${1:-captaincook4d_do}"
CHECKPOINT_DIR="${2:-checkpoints/}"
MIN_SEEDS="${3:-5}"

# Run evaluation
python3 -c "
import sys
sys.path.insert(0, '/home/user')
from eval.evaluator import evaluate_all_scenarios, write_scores_json
from data.loader import load_data
import json

# Load data
repo_path = '/home/user/method/repo/data/captaincook4d'
data = load_data(repo_data_path=repo_path)
print(f'Loaded {len(data)} scenarios')

if '$METHOD_NAME' == 'captaincook4d_partial_compare':
    do_results = evaluate_all_scenarios(
        data,
        do_predictions_dir='$CHECKPOINT_DIR',
        method='do',
        allow_partial=True,
        min_do_seeds=int('$MIN_SEEDS'),
    )
    completed = set(do_results['per_scenario'].keys())
    if not completed:
        raise SystemExit('No scenarios with sufficient corrected DO seeds are available yet.')
    subset_data = [sd for sd in data if sd.scenario_name in completed]
    count_results = evaluate_all_scenarios(subset_data, method='count_based')
    print(f'Completed subset: {len(completed)} scenarios -> {sorted(completed)}')
    print(f'Partial Count-Based: F1={count_results[\"aggregate\"][\"mean_f1\"]:.1f}')
    print(f'Partial DO: F1={do_results[\"aggregate\"][\"mean_f1\"]:.1f}')
    write_scores_json(
        count_results,
        'captaincook4d_count_partial',
        dataset_override='captaincook4d_task_graph',
        method_override='count_based',
    )
    write_scores_json(
        do_results,
        'captaincook4d_do_partial',
        dataset_override='captaincook4d_task_graph',
        method_override='do',
    )
elif 'count' in '$METHOD_NAME':
    # Count-based baseline
    results = evaluate_all_scenarios(data, method='count_based')
    print(f'Count-Based: F1={results[\"aggregate\"][\"mean_f1\"]:.1f}')
    write_scores_json(results, '$METHOD_NAME')
else:
    if '$CHECKPOINT_DIR' == '' or '$CHECKPOINT_DIR' == 'checkpoints/':
        raise SystemExit('DO evaluation requires an explicit checkpoint/results directory.')
    results = evaluate_all_scenarios(data, do_predictions_dir='$CHECKPOINT_DIR', method='do')
    print(f'DO: F1={results[\"aggregate\"][\"mean_f1\"]:.1f}')
    write_scores_json(results, '$METHOD_NAME')

print('Scores written to scoring/scores.json')
"

echo "Evaluation complete."
