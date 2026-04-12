#!/bin/bash
# Paper-faithful CaptainCook4D reproduction job.
# Runs the count-based baseline, then fixed-epoch DO training with 5 seeds.
set -e
cd /home/user
echo "=== $(date) === CaptainCook4D baseline + DO reproduction ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

bash scripts/baseline.sh
METHOD_DEVICE=cpu METHOD_SEEDS=42,1337,2024,2025,2026 METHOD_MAX_EPOCHS=1000 \
METHOD_OUTPUT_DIR=checkpoints/do_captaincook4d_fixedtgml \
bash scripts/method.sh
python3 validate.py --compare
echo "=== Done ==="
