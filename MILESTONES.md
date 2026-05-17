# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-17 17:28] - core_claim
- Verified that the implementation is the paper's TGML Direct Optimization algorithm.
- Completed a full 24-procedure CaptainCook4D rerun with 5 DO seeds per procedure.
- Reproduced the core claim: DO F1 76.36 beats Count-Based F1 61.41 on the full benchmark.

### [2026-05-17 18:05] - core_claim_plus
- Added and refreshed train/test split score files as additional fixed evaluation settings.
- Corrected evaluator thresholding to the official `1/n_gt` rule and refreshed all score files.
- Added constraint and structural metrics to the main `scores.json`.

## Stop Justification
- Completed at milestone `core_claim_plus`: the paper's actual TGML/DO method runs end-to-end, supports the central DO-over-Count-Based claim on CaptainCook4D, and is also checked on train/test procedure splits.
