#!/bin/bash
# Reproduce the reference end to end (CPU, ~15 min): train the paper's DO model
# (TGML loss, SA early stopping, 5 seeds) on every procedure of both scored
# experiments, then score it with the public and hidden evaluators.
# The hardening audit (variants and shortcuts) is notes/hardening/run_hardening.sh.
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

echo "=== $(date) === reference: DO + TGML + SA early stopping, 5 seeds, both experiments ==="
rm -rf checkpoints/do
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python3 validate.py --compare --post-design
echo "=== $(date) === done ==="
