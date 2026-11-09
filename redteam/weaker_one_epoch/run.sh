#!/bin/bash
set -euo pipefail
cd /home/user
source scripts/common_env.sh
python3 -m redteam.execute_case weaker_one_epoch
