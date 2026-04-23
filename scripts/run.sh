#!/bin/bash
# Generate train and test evaluation scores.
set -e
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === Train/Test Evaluation ==="

# Run train slice evaluation
echo "Running train slice evaluation..."
bash scripts/evaluate_train.sh

# Run test slice evaluation
echo "Running test slice evaluation..."
bash scripts/evaluate_test.sh

# Verify outputs exist
echo "Checking outputs..."
for f in scoring/scores_train.json scoring/scores_test.json; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
    echo "$f exists"
done

# Verify slice markers
python3 - <<'PY'
import json

for path in ["scoring/scores_train.json", "scoring/scores_test.json"]:
    with open(path) as f:
        scores = json.load(f)
    slice_val = scores.get("slice", "MISSING")
    print(f"{path}: slice={slice_val}")
    if slice_val == "MISSING":
        raise SystemExit(f"Missing slice marker in {path}")
    if slice_val not in ("train", "test"):
        raise SystemExit(f"Invalid slice value in {path}: {slice_val}")

# Verify method coverage matches
train_scores = json.load(open("scoring/scores_train.json"))
test_scores = json.load(open("scoring/scores_test.json"))

train_methods = set(train_scores["experiments"]["captaincook4d_task_graph"]["results"].keys())
test_methods = set(test_scores["experiments"]["captaincook4d_task_graph"]["results"].keys())

if train_methods != test_methods:
    print(f"WARNING: Method mismatch - train: {train_methods}, test: {test_methods}")
else:
    print(f"Method coverage matches: {train_methods}")

print("All checks passed!")
PY

echo "=== Done ==="
