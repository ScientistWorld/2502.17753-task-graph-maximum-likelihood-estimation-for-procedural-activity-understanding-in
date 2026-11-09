#!/bin/bash
set -euo pipefail
cd /home/user
source scripts/common_env.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p redteam
python3 -m redteam.train_variants 2>&1 | tee redteam/training.log
for case in reference_saved reference_rerun reference_seed_7 reference_seed_19 reference_seed_31 better_more_training better_sequence_selection weaker_one_epoch simple_baseline brute_force shortcut_visible_labels shortcut_seed_ids shortcut_partial_compare shortcut_forge_scores shortcut_rewrite_ground_truth shortcut_replace_baseline shortcut_zero shortcut_ones shortcut_nan shortcut_infinity shortcut_negative shortcut_empty; do
    bash "redteam/$case/run.sh" 2>&1 | tee "redteam/$case/run.log"
done
python3 -m redteam.analyze
