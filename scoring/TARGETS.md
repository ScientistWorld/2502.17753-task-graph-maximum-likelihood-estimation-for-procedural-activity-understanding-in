# Evaluation Targets

## Experiment: `captaincook4d_task_graph`
This workspace is currently scoped to the paper's CaptainCook4D task-graph generation benchmark, averaged over 24 cooking procedures.

## Primary Target
- **Metric**: `f1`
- **Goal**: reproduce the paper's central claim that Direct Optimization (DO) substantially improves edge-prediction F1 over prior approaches on CaptainCook4D.

## Constraint Target
- **Metric**: `std_f1`
- **Goal**: keep run-to-run variability low, reflecting the paper's claim that DO is stable across seeds.

## Required Comparison
- **Compared methods in `scores.json`**: `count_based`, `do`
- **Claim checked**: `do.f1` should clearly exceed `count_based.f1`, while `do.recall` should remain strong rather than winning only through conservative edge prediction.

## How To Run
- `bash scripts/baseline.sh`
- `bash scripts/method.sh`
- `python validate.py --compare`
