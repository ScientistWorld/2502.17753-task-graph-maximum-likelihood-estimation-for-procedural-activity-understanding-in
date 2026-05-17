# Evaluation Targets

## Experiment: `captaincook4d_task_graph`
This workspace is anchored to the CaptainCook4D task-graph generation benchmark. The current reproduced result is a corrected full rerun on all 24 procedures with complete saved predictions for all required seeds.

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

## Structural Constraint Targets
- **Metric**: `edge_density_ratio`
- **Goal**: avoid improving recall by returning graphs that are much denser than the ground truth.

- **Metric**: `path_coverage`
- **Goal**: preserve useful directed paths in the predicted DAG, not just isolated edge overlap.

- **Metric**: `topological_distance`
- **Goal**: keep the predicted partial order close to the ground-truth partial order. Lower is better.

## Required Comparison
- **Compared methods in `scores.json`**: `count_based`, `do`
- **Claim checked**: the stronger learned method should beat the simple counting baseline by a clear margin on F1 while staying competitive on both precision and recall.

## Current Status
- The present `scores.json` reflects the full corrected 24-procedure rerun. Future improvements should keep the same scoring format and preserve full-procedure coverage.

## How To Run
- `bash scripts/baseline.sh`
- `bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d_fixedtgml`
- `bash scripts/evaluate_train.sh checkpoints/do_captaincook4d_fixedtgml`
- `bash scripts/evaluate_test.sh checkpoints/do_captaincook4d_fixedtgml`
- `python validate.py`
