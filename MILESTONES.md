# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-17 17:26] - core_claim_plus
- Job `022a3108-95f` completed the corrected CaptainCook4D rerun for all 24 procedures with 5 seeds each
- Full-benchmark `scores.json`: DO F1=76.36 vs count_based F1=61.41; precision and recall both remain higher for DO
- `validate.py --compare` passed after the full rerun, and `scoring/scores.json` now reflects the full benchmark rather than a partial subset
- Stopped at `core_claim_plus`: the paper's central TGML/DO claim is reproduced across the full primary CaptainCook4D benchmark, but not all secondary paper baselines or alternative method rows were reimplemented

### [2026-05-17 16:45] - core_claim_plus
- Job `6a0a2cb2-040` completed successfully and expanded corrected CaptainCook4D coverage from 10 to 14 procedures
- New completed procedures: `Tomatomozzarellasalad`, `Microwavemugpizza`, `Panfriedtofu`, `Cucumberraita`
- Updated `scores.json`: DO F1=76.07 vs count_based F1=63.13 on the 14-procedure corrected subset
- Prepared the next CPU continuation to attempt the remaining 10 procedures in a single resumable normal-queue job

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
- Stopped after reaching `core_claim_plus`. The actual TGML Direct Optimization method now runs on the full 24-procedure CaptainCook4D benchmark with the paper's 5-seed / 1000-epoch setting and clearly beats the count-based baseline. Remaining unreproduced items are secondary comparison rows from the paper, not blockers for the core gym.
