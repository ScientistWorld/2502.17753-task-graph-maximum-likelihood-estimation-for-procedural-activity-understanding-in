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
# 1 seed, 300 epochs (matching paper's evaluation point for Coffee scenario F1=87.8).
# Training outputs NPZ files with pred_cores per scenario.
METHOD_MAX_EPOCHS=300 METHOD_SEEDS=42 METHOD_DEVICE=cuda bash scripts/method.sh

# Evaluate DO predictions against ground truth graphs.
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d/

# Validate the fresh scores against the reference.
python3 validate.py --compare
