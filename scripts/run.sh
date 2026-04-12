#!/bin/bash
# Paper-faithful CaptainCook4D reproduction job.
# Runs the count-based baseline, then fixed-epoch DO training with 5 seeds.
set -e
cd /home/user
source /home/user/scripts/common_env.sh
echo "=== $(date) === CaptainCook4D baseline + DO reproduction ==="
python3 - <<'PY'
import os
import site
import sys
import numpy
import networkx
import torch
print('PYTHONNOUSERSITE:', os.environ.get('PYTHONNOUSERSITE'))
print('ENABLE_USER_SITE:', site.ENABLE_USER_SITE)
print('sys.path:')
for path in sys.path:
    print('  ', path)
if any(path.startswith('/home/user/.local') for path in sys.path):
    raise SystemExit('Refusing to run: /home/user/.local is present on sys.path')
print('numpy:', numpy.__version__, numpy.__file__)
print('networkx:', networkx.__version__, networkx.__file__)
print('torch:', torch.__version__, torch.__file__)
print('CUDA:', torch.cuda.is_available())
PY

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

if [ -d checkpoints/do_captaincook4d_fixedtgml ]; then
  python3 <<'PY'
import json
import os
import numpy as np

scores_path = "scoring/scores.json"
pred_dir = "checkpoints/do_captaincook4d_fixedtgml"
complete = 0
if os.path.isdir(pred_dir):
    for fn in os.listdir(pred_dir):
        if not fn.endswith("_predictions.npz"):
            continue
        data = np.load(os.path.join(pred_dir, fn))
        if len(data["pred_cores"]) >= 5:
            complete += 1

if os.path.exists(scores_path):
    with open(scores_path) as f:
        scores = json.load(f)
    exp = scores.get("experiments", {}).get("captaincook4d_task_graph", {})
    results = exp.get("results", {})
    if complete < 24 and "do" in results:
        print(f"Removing stale full-dataset DO score; only {complete}/24 corrected scenarios are complete.")
        del results["do"]
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
PY
  bash scripts/evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5 || true
fi

METHOD_DEVICE=cpu METHOD_SEEDS=42,1337,2024,2025,2026 METHOD_MAX_EPOCHS=1000 \
METHOD_OUTPUT_DIR=checkpoints/do_captaincook4d_fixedtgml METHOD_SCENARIOS=cheesepimiento METHOD_MAX_SCENARIOS=1 \
bash scripts/method.sh
bash scripts/evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5 || true
python3 validate.py
echo "=== Done ==="
