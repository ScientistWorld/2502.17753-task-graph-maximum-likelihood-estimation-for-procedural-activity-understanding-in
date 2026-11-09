#!/bin/bash
# The Count-Based baseline is computed by the evaluators themselves (fixed by
# eval/), so scoring anything also scores the baseline under the key "count_based".
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh
bash scripts/evaluate.sh
