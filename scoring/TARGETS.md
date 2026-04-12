# Evaluation Targets

## Experiment: `captaincook4d_task_graph`
This workspace is still anchored to the paper's CaptainCook4D task-graph generation benchmark. The current minimum reproducible target compares the corrected Direct Optimization (DO) checkpoints against the count-based baseline on the subset of scenarios that already have all 5 post-fix DO seeds saved in `checkpoints/do_captaincook4d_fixedtgml/`.

## Primary Target
- **Metric**: `f1`
- **Goal**: reproduce the paper's central claim that DO substantially improves edge-prediction F1 over the count-based baseline using the paper's actual TGML + DO computation and corrected post-fix checkpoints.

## Constraint Target
- **Metric**: `std_f1`
- **Goal**: keep run-to-run variability low on the completed subset, reflecting the paper's claim that DO is stable across seeds.

## Required Comparison
- **Compared methods in `scores.json`**: `count_based`, `do`
- **Claim checked**: `do.f1` should clearly exceed `count_based.f1`, while `do.recall` remains strong. At the current stage these rows are sourced from the corrected completed subset; the full 24-scenario DO result remains the next expansion target after this minimum claim.

## How To Run
- `bash scripts/baseline.sh`
- `bash scripts/evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5`
- `python validate.py`
