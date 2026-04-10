#!/bin/bash
# Job script submitted via action.yaml.
# Runs inside the compute container with GPU(s) available.

set -e
cd /home/user

# Sanity check: verify core dependencies are available.
python3 -c "
import torch, numpy, networkx, json
print('runtime sanity:', torch.__version__, numpy.__version__, networkx.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU detected, using CPU')
"

# Remove stale outputs so this submission only reports fresh results.
rm -f scoring/scores.json

# Remove previous DO checkpoints so this is a clean run.
rm -rf checkpoints/do_captaincook4d

# Run the DO method with paper-specified settings:
# 5 seeds (matching paper's std_f1 computation), 1000 epochs max, lr=0.1, beta=0.005.
# The training loop outputs sequence accuracy (weak supervision) but evaluation uses the
# full adjacency post-processing pipeline from eval/ to compute edge-level F1.
METHOD_MAX_EPOCHS=1000 METHOD_SEEDS="42 43 44 45 46" METHOD_DEVICE=cpu bash scripts/method.sh

# Evaluate all scenarios (baseline + DO) and write scores.json.
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d/
bash scripts/evaluate.sh captaincook4d_count

# Validate the fresh scores against the reference.
python3 validate.py --compare
