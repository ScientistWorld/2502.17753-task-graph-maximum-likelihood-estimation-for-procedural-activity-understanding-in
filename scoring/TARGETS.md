# Evaluation Targets

## Experiment: `captaincook4d_task_graph_partial_completed`
This is the current minimum reproducible target: compare the corrected Direct Optimization (DO) checkpoints against the count-based baseline on the subset of CaptainCook4D scenarios that already have all 5 post-fix DO seeds saved in `checkpoints/do_captaincook4d_fixedtgml/`.

## Primary Target
- **Metric**: `f1`
- **Goal**: reproduce the paper's central claim that DO substantially improves edge-prediction F1 over the count-based baseline using the paper's actual TGML + DO computation and corrected post-fix checkpoints.

## Constraint Target
- **Metric**: `std_f1`
- **Goal**: keep run-to-run variability low on the completed subset, reflecting the paper's claim that DO is stable across seeds.

## Required Comparison
- **Compared methods in `scores.json`**: `count_based`, `do`
- **Claim checked**: `do.f1` should clearly exceed `count_based.f1`, while `do.recall` remains strong. The full 24-scenario CaptainCook4D benchmark remains the next expansion target after this minimum claim.

## How To Run
- `bash scripts/baseline.sh`
- `bash scripts/evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5`
- `python validate.py`
