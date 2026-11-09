#!/bin/bash
set -euo pipefail
cd /home/user
source scripts/common_env.sh
bash redteam/with_prerequisite.sh python3 -m redteam.execute_case shortcut_partial_compare
