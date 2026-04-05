#!/bin/bash
# Job script submitted via action.yaml.
# Runs inside the compute container with GPU(s) available.

set -e
cd /home/user

# Remove stale outputs so this submission only reports fresh results.
rm -f scoring/scores.json

# Run the minimum end-to-end reproduction for the CaptainCook4D core claim.
bash scripts/reproduce.sh

# Validate the fresh scores against the scoped reference experiment.
python3 validate.py --compare
