#!/bin/bash
# Job script submitted via action.yaml.
# Runs inside the compute container with GPU(s) available.

set -e
cd /home/user

# Let PyTorch auto-detect GPU if available (no manual CUDA_VISIBLE_DEVICES)
export METHOD_MAX_EPOCHS="${METHOD_MAX_EPOCHS:-1000}"
export METHOD_SEEDS="${METHOD_SEEDS:-42 43 44 45 46}"

python3 - <<'PY'
import torch
import numpy
import networkx
print("runtime sanity:", torch.__version__, numpy.__version__, networkx.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
else:
    print("No GPU detected, using CPU")
PY

# Remove stale outputs so this submission only reports fresh results.
rm -f scoring/scores.json

# Run the minimum end-to-end reproduction for the CaptainCook4D core claim.
bash scripts/reproduce.sh

# Validate the fresh scores against the scoped reference experiment.
python3 validate.py --compare
