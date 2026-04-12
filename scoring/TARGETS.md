# Evaluation Targets

## Experiment: `captaincook4d_task_graph`
This workspace is anchored to the CaptainCook4D task-graph generation benchmark. The current reproduced result is a corrected partial rerun on the subset of procedures that already have complete saved predictions for all required seeds.

## Primary Target
- **Metric**: `f1`
- **Goal**: improve graph-edge recovery quality on CaptainCook4D. This is the main score to optimize.

## Secondary Quality Signals
- **Metric**: `precision`
- **Goal**: avoid improving F1 by adding too many spurious edges.

- **Metric**: `recall`
- **Goal**: preserve coverage of true prerequisite edges instead of optimizing for a high-precision but incomplete graph.

## Constraint Target
- **Metric**: `std_f1`
- **Goal**: keep seed-to-seed variability low. A method that improves mean F1 only by becoming unstable is less useful.

## Required Comparison
- **Compared methods in `scores.json`**: `count_based`, `do`
- **Claim checked**: the stronger learned method should beat the simple counting baseline by a clear margin on F1 while staying competitive on both precision and recall.

## Current Limitation
- The present `scores.json` reflects the corrected completed subset rather than a full 24-procedure rerun. Future improvements should keep the same scoring format and ideally expand coverage to the full benchmark.

## How To Run
- `bash scripts/baseline.sh`
- `bash scripts/evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5`
- `python validate.py`
