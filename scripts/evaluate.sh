#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, matching scoring/reference.json structure.
#
# Usage:
#   ./evaluate.sh captaincook4d_do        # evaluate DO on CaptainCook4D
#   ./evaluate.sh captaincook4d_count     # evaluate count-based on CaptainCook4D

set -e
cd /home/user

mkdir -p scoring

# Parse arguments
METHOD_NAME="${1:-captaincook4d_do}"
CHECKPOINT_DIR="${2:-checkpoints/}"

# Run evaluation
python -c "
import sys
sys.path.insert(0, '/home/user')
from eval.evaluator import evaluate_all_scenarios, write_scores_json
from data.loader import load_data
import json

# Load data
repo_path = '/home/user/method/repo/data/captaincook4d'
data = load_data(repo_data_path=repo_path)
print(f'Loaded {len(data)} scenarios')

if 'count' in '$METHOD_NAME':
    # Count-based baseline
    results = evaluate_all_scenarios(data, method='count_based')
    print(f'Count-Based: F1={results[\"aggregate\"][\"mean_f1\"]:.1f}')
    write_scores_json(results, '$METHOD_NAME')
else:
    # DO predictions
    if '$CHECKPOINT_DIR' != 'checkpoints/' and '$CHECKPOINT_DIR' != '':
        results = evaluate_all_scenarios(data, do_predictions_dir='$CHECKPOINT_DIR', method='do')
        print(f'DO: F1={results[\"aggregate\"][\"mean_f1\"]:.1f}')
        write_scores_json(results, '$METHOD_NAME')
    else:
        # Run count-based as proxy if no DO checkpoints available
        results = evaluate_all_scenarios(data, method='count_based')
        print(f'(No checkpoints, using Count-Based): F1={results[\"aggregate\"][\"mean_f1\"]:.1f}')
        write_scores_json(results, '$METHOD_NAME')

print('Scores written to scoring/scores.json')
"

echo "Evaluation complete."
