> **Superseded (final hardening iteration):** the evaluation contract, splits, metrics and reference described below were revised. See `HARDENING.md`, `briefing/evaluation.md` and `scoring/TARGETS.md` for the current gym.

# Verification Report

## Initial Assessment

The workspace entered final verification at `core_claim_plus`. That milestone is justified: the implementation runs the paper's TGML Direct Optimization method on CaptainCook4D, evaluates saved prediction artifacts, and reports a clear full-benchmark F1 gain over the Count-Based baseline. The train/test split artifacts are present and calibrated, with matching method keys and comparable difficulty.

Problem/solution separation is clean. `briefing/problem.md`, `briefing/evaluation.md`, and `scoring/EXPERIMENTS.md` describe the task and scoring without leaking the paper's method. `eval/` and `data/` do not import from `method/`, and the evaluator reads method outputs instead of regenerating the method.

## Issues Found

- **Minor**: `scripts/evaluate.sh` did not satisfy the standalone evaluation contract with no arguments; it defaulted to a DO evaluation without a concrete prediction directory and exited.
- **Cosmetic**: `PROGRESS.md` contained a stale data-path description from an earlier layout.
- **Pre-existing workspace noise**: git status still shows harness/system artifacts outside the gym code path (`README.md`, `manifest.json`, deleted `trajectory/*`). These were not created by this verification pass and were left untouched.

## Fixes Applied

- Reworked `scripts/evaluate.sh` so no-argument evaluation enumerates methods from `scoring/scores.json` or complete prediction directories, scores the built-in count baseline and prediction-artifact methods, and still supports explicit method/directory calls.
- Verified the download source in `scripts/download.sh` with `curl -I -L`; the GitHub repository resolves to HTTP 200 after redirect.
- Removed generated `__pycache__` directories from gym code directories.
- Updated `PROGRESS.md` with the portable `data/downloads/` data path and final verification notes.
- Re-ran `bash scripts/evaluate.sh` and `python3 validate.py --compare --post-design`; all checks passed.

## Final Judgment

This workspace is ready to serve as a research gym for the scoped CaptainCook4D task-graph generation claim. Improve-mode agents can build on the packaged TGML implementation and saved results; from-scratch agents can use the method-agnostic data, problem definition, and reusable evaluation interface without seeing solution details.

The main limitation is coverage: the gym does not reproduce EgoPER, EgoProceL, TGT text/video, or downstream task results. Within the declared CaptainCook4D scope, the scoring is usable and discriminative.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: scoped to CaptainCook4D task-graph generation only
