#!/bin/bash
set -euo pipefail
cd /home/user
source scripts/common_env.sh
python3 redteam/check_stock_runtime.py
bash redteam/shortcut_single_seed/run.sh 2>&1 | tee redteam/shortcut_single_seed/run.log
bash redteam/with_prerequisite.sh python3 -m redteam.analyze
