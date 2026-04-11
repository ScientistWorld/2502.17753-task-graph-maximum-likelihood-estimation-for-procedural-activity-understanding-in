#!/bin/bash
# Job script submitted via action.yaml.
# Runs inside the compute container with CPU cores for parallel scenario training.

set -e
cd /home/user

# Sanity check: verify core dependencies are available.
python3 -c "
import torch, numpy, networkx, json
print('runtime sanity:', torch.__version__, numpy.__version__, networkx.__version__)
"

# Remove stale scores so this submission only reports fresh results.
rm -f scoring/scores.json

# Run the DO method with paper-specified settings:
# 24 parallel workers (one per scenario), 1 seed, 300 epochs.
# Parallel training across 24 CPU cores brings total runtime from ~17min to ~1min.
# Evaluate predictions against ground truth graphs.
METHOD_MAX_EPOCHS=300 METHOD_WORKERS=24 bash scripts/method.sh

# Validate the fresh scores against the reference.
python3 validate.py --compare
