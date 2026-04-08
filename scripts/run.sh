#!/bin/bash
# Job script submitted via action.yaml.
# Runs inside the compute container with GPU(s) available.

set -e
cd /home/user

export METHOD_DEVICE="${METHOD_DEVICE:-cpu}"
export METHOD_MAX_EPOCHS="${METHOD_MAX_EPOCHS:-1000}"
export METHOD_SEEDS="${METHOD_SEEDS:-42}"

python3 - <<'PY'
import torch
import numpy
import networkx
print("runtime sanity:", torch.__version__, numpy.__version__, networkx.__version__)
PY

# Remove stale outputs so this submission only reports fresh results.
rm -f scoring/scores.json

# Run the minimum end-to-end reproduction for the CaptainCook4D core claim.
bash scripts/reproduce.sh

# Validate the fresh scores against the scoped reference experiment.
python3 validate.py --compare
