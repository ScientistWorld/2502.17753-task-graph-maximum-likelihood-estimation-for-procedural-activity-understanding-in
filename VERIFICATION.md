# Verification Report

## Initial Assessment
The workspace started in a usable `core_claim` state: the CaptainCook4D reproduction code, evaluation pipeline, and partial corrected scores were coherent, `validate.py` passed, and `scripts/evaluate.sh` could still regenerate the current `scores.json`. The main problems were gym-facing rather than algorithmic: the workspace root contained many scratch files and logs, and several problem/scoring documents leaked the reproduced method in files that should stay method-agnostic.

## Issues Found
- Critical: The workspace root contained many raw scratch scripts, text dumps, and log files that did not belong at the top level of a reusable gym.
- Critical: `briefing/problem.md`, `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, `scoring/CONSTRAINTS.md`, and `scoring/DIRECTION.md` leaked method details that from-scratch scientist agents should not see.
- Minor: `scoring/reference.json` over-weighted precision and recall relative to the primary metric `f1`, which made the gym rubric easier to game than necessary.

## Fixes Applied
- Moved root-level scratch artifacts and ad hoc logs into `tmp/root_artifacts/` so the workspace root now follows the expected research-gym layout.
- Rewrote the method-agnostic briefing and scoring guidance documents to describe the task, benchmark, scoring targets, and constraints without exposing architecture, loss-design, or training-recipe details from the reproduced method.
- Adjusted `scoring/reference.json` coefficients so `f1` is clearly the dominant scored metric, with precision and recall treated as secondary signals and `std_f1` as a modest stability constraint.
- Re-ran `python validate.py --compare` and `bash scripts/evaluate.sh captaincook4d_partial_compare checkpoints/do_captaincook4d_fixedtgml 5` to confirm the cleanup did not break scoring.

## Final Judgment
This workspace is now suitable as a research gym at the current milestone. A scientist agent can use the problem definition and evaluation stack without being handed the reproduced algorithm, and improve-mode agents still have the full method and checkpoint context. The remaining limitation is that the scored reproduction is still based on the corrected completed CaptainCook4D subset rather than a full 24-procedure rerun, so the gym is ready but not yet fully expanded.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: `scores.json` still reflects the corrected 8-procedure CaptainCook4D subset rather than a full 24-procedure rerun
