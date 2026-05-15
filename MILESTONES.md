# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-15] - core_claim (designer iteration complete)
- Enhanced train/test evaluation with structural metrics (edge_density_ratio, path_coverage, topological_distance)
- Train/test split: 13 train scenarios, 12 test scenarios
- DO method scores: train F1=79.54, test F1=79.59
- Count-based baseline scores: train F1=61.43, test F1=61.40
- Small expected gap confirms calibrated split
- Structural metrics added to detect metric gaming:
  - edge_density_ratio: catches over/under-connection
  - path_coverage: measures structural path correctness
  - topological_distance: rewards correct dependency ordering

### [2026-04-23] - core_claim (initial designer iteration)
- Train/test split for CaptainCook4D task graph benchmark
- eval/train/ (13 scenarios) and eval/test/ (12 scenarios) with auto-discovery
- Scripts auto-discover all methods and add slice markers
- DO outperforms count_based on both slices: ~80 F1 vs ~61 F1

### [2026-04-12] - core_claim
- Completed CaptainCook4D full-dataset rerun with corrected checkpoints
- DO method: 24 scenarios complete, all 5 seeds each
- DO F1=78.0 vs count-based F1=67.2 on full 24-scenario dataset
- Core claim confirmed: Direct Optimization beats count-based baseline

### [2026-04-12] - method_runs
- Fixed DO training pipeline with correct auxiliary token indexing
- Implemented TGML (Task Graph Marker) with direct optimization
- Training runs end-to-end without errors
- Count-based baseline evaluates correctly

### [2026-04-12] - none
- Workspace inherited with partial CaptainCook4D reproduction
- Paper: "Direct Optimization of Task Graph Loss" (CaptainCook4D)
- Core claim: DO outperforms count-based baseline on task graph generation

## Stop Justification
- Completed at milestone `core_claim`.
