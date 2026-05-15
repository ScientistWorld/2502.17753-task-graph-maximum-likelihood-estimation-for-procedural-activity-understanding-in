# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-15] - core_claim
- Workspace inherited with complete DO+TGML implementation on CaptainCook4D
- scores.json: DO F1=77.98 vs count_based F1=67.20 — core claim confirmed
- 24/24 scenario prediction files in checkpoints/do_captaincook4d/
- Train/test split with structural metrics (edge_density_ratio, path_coverage, topological_distance)
- Validation passes all structural checks
- DO outperforms count_based on both train (79.5 vs 61.4) and test (79.6 vs 61.4) slices

## Stop Justification
- Completed at milestone `method_runs`.
