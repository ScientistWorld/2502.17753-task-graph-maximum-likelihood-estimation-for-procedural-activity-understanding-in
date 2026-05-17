# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-17 16:50] - core_claim
- Job `b6f3325a-6ce` completed successfully and expanded corrected CaptainCook4D coverage from 8 to 10 procedures
- New completed procedures: `Microwavefrenchtoast`, `Capresebruschetta`
- Updated `scores.json`: DO F1=77.42 vs count_based F1=64.03 on the 10-procedure corrected subset
- Fixed noisy scenario-count parsing in `scripts/method.sh` and prepared a larger 4-procedure CPU continuation batch

### [2026-05-15 16:18] - core_claim
- Audited reset workspace from `method_runs` and restored `core_claim`
- Confirmed `method/` implements the actual TGML Direct Optimization algorithm, not a surrogate
- Confirmed `eval/` is method-independent and standalone partial evaluation reproduces DO F1=77.98 vs count_based F1=67.20 on 8 corrected CaptainCook4D scenarios
- Kept `action.yaml` as a CPU continuation submission to expand corrected coverage

### [2026-05-15 16:14] - core_claim
- Revalidated current corrected CaptainCook4D subset with standalone evaluator
- `scores.json`: DO F1=77.98 vs count_based F1=67.20 on 8 complete corrected scenarios
- Prepared a CPU continuation job to expand corrected coverage beyond the current subset

### [2026-05-15] - core_claim
- Workspace inherited with complete DO+TGML implementation on CaptainCook4D
- scores.json: DO F1=77.98 vs count_based F1=67.20 — core claim confirmed
- 24/24 scenario prediction files in checkpoints/do_captaincook4d/
- Train/test split with structural metrics (edge_density_ratio, path_coverage, topological_distance)
- Validation passes all structural checks
- DO outperforms count_based on both train (79.5 vs 61.4) and test (79.6 vs 61.4) slices

## Stop Justification
- Not stopped. Current milestone is `core_claim`; the next job continues the corrected CaptainCook4D rerun to expand coverage.
