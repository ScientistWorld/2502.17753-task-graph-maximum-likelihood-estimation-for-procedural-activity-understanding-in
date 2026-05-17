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

### [2026-05-17 18:06] - core_claim_plus
- Continuation audit restored `core_claim_plus`: method code implements TGML Direct Optimization, not a surrogate.
- Re-ran standalone full, train-slice, and test-slice evaluation scripts; all completed successfully.
- Re-ran `python3 validate.py --compare`; all scoring and structure checks passed.

### [2026-05-17 18:22] - core_claim_plus
- Added an opt-in DO sequence-selection mode matching the paper's weakly supervised sequence-accuracy early-stopping signal.
- Prepared a fresh CaptainCook4D rerun in `checkpoints/do_captaincook4d_seqselect/` to test whether this closes the remaining gap to the paper's reported DO result.

### [2026-05-17 18:30] - core_claim_plus
- Completed final gym verification.
- Confirmed method-agnostic briefing/evaluation files, eval-method separation, download portability, and train/test scoring checks.
- Fixed the main score file to include `worst_quartile_f1`, matching the nonzero metric in `reference.json`.

## Stop Justification
- Final verification completed at milestone `core_claim_plus`; the gym is ready for the scoped CaptainCook4D task-graph generation benchmark.
