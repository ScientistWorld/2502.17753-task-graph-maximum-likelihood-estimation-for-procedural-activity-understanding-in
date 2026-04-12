#!/bin/bash
# Reproduce the paper's results end-to-end.
# Run this inside the compute container after running scripts/download.sh.

set -e
cd /home/user
export PYTHONPATH="/home/user/environment/pylibs${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Reproducing TGML Paper Results ==="

# Clean slate
rm -f scoring/scores.json

# Step 1: Count-Based baseline
echo "[1/2] Running Count-Based baseline..."
bash scripts/baseline.sh

# Step 2: DO method
echo "[2/2] Running DO method (5 seeds, 1000 epochs each)..."
METHOD_DEVICE=cpu METHOD_SEEDS=42,1337,2024,2025,2026 METHOD_MAX_EPOCHS=1000 \
METHOD_OUTPUT_DIR=checkpoints/do_captaincook4d_fixedtgml \
bash scripts/method.sh

if [ ! -f scoring/scores.json ] || ! python3 - <<'PY'
import json
import os

path = '/home/user/scoring/scores.json'
if not os.path.exists(path):
    raise SystemExit(1)
data = json.load(open(path))
do = data.get('experiments', {}).get('captaincook4d_task_graph', {}).get('results', {}).get('do')
raise SystemExit(0 if do else 1)
PY
then
  bash scripts/evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5 || true
fi

echo ""
echo "=== Reproduction complete ==="
echo "Results in scoring/scores.json"
python3 validate.py
