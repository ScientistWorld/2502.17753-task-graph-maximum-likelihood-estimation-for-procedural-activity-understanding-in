#!/bin/bash
# Run the Count-Based baseline method.
#
# Usage:
#   bash scripts/baseline.sh

set -e
cd /home/user
export PYTHONPATH="/home/user/environment/pylibs${PYTHONPATH:+:$PYTHONPATH}"

echo "Running Count-Based baseline on CaptainCook4D..."

# Run evaluation with count-based
bash scripts/evaluate.sh captaincook4d_count

echo "Done."
