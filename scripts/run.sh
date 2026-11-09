#!/bin/bash
# Job: reproduce the reference end to end (CPU) on both scored experiments
# (CaptainCook4D and the noisy-demonstration benchmark in data/noisy_demos), run the
# hardening audit, regenerate the reference-only score files.
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === environment ==="
python3 -c "import sys, numpy, networkx, torch; print(sys.version); print('numpy', numpy.__version__); print('networkx', networkx.__version__, networkx.__file__); print('torch', torch.__version__)"

PREV="$(mktemp -d)"
cp checkpoints/do/*_predictions.npz "$PREV"/ 2>/dev/null || true

echo "=== $(date) === reference: DO + TGML + SA early stopping, 5 seeds, both experiments ==="
rm -rf checkpoints/do
bash scripts/method.sh
python3 - "$PREV" <<'PY'
import glob, os, sys
import numpy as np
prev = sys.argv[1]
same, diff = 0, []
for path in sorted(glob.glob(os.path.join(prev, "*_predictions.npz"))):
    name = os.path.basename(path)
    if name.startswith("noisy"):
        continue  # the noisy benchmark was regenerated (enlarged)
    a = np.load(path)["pred_cores"]
    b = np.load(os.path.join("checkpoints/do", name))["pred_cores"]
    if a.shape == b.shape and np.array_equal(a, b):
        same += 1
    else:
        diff.append(name)
print(f"reference vs previous job (same procedures): {same} identical, differing: {diff}")
PY
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

echo "=== $(date) === hardening audit ==="
bash notes/hardening/run_hardening.sh

echo "=== $(date) === final reference-only score files ==="
ls checkpoints
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python3 notes/hardening/analyze.py
python3 validate.py --compare --post-design || echo "validate.py reported errors (see above)"
echo "=== $(date) === done ==="
