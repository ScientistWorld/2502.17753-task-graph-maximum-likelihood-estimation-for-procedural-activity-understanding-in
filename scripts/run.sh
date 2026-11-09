#!/bin/bash
# Job: reproduce the reference end to end (CPU), run the hardening audit, then
# regenerate the reference-only score files.
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === environment ==="
python3 -c "import sys, numpy, networkx, torch; print(sys.version); print('numpy', numpy.__version__); print('networkx', networkx.__version__, networkx.__file__); print('torch', torch.__version__)"

echo "=== $(date) === reference: DO + TGML + SA early stopping, 5 seeds, 24 procedures ==="
rm -rf checkpoints/do
bash scripts/method.sh
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
