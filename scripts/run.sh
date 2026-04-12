#!/bin/bash
# Paper-faithful CaptainCook4D reproduction job.
# Runs the count-based baseline, then fixed-epoch DO training with 5 seeds.
set -e
cd /home/user
echo "=== $(date) === CaptainCook4D baseline + DO reproduction ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

if python3 <<'PY'
import json
import os

path = "scoring/scores.json"
need_baseline = True
if os.path.exists(path):
    try:
        scores = json.load(open(path))
        need_baseline = "count_based" not in scores.get("experiments", {}).get("captaincook4d_task_graph", {}).get("results", {})
    except Exception:
        need_baseline = True
print("Need baseline:", need_baseline)
raise SystemExit(0 if need_baseline else 1)
PY
then
  bash scripts/baseline.sh
else
  echo "Skipping baseline; count_based score already present."
fi

METHOD_DEVICE=cpu METHOD_SEEDS=42,1337,2024,2025,2026 METHOD_MAX_EPOCHS=1000 \
METHOD_OUTPUT_DIR=checkpoints/do_captaincook4d_fixedtgml \
bash scripts/method.sh
python3 validate.py --compare
echo "=== Done ==="
