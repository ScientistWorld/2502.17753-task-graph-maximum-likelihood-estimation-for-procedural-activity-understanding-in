# Verification Report

## Initial Assessment

The workspace entered final verification at `core_claim_plus`. That milestone is justified: the implementation runs the paper's TGML Direct Optimization method on CaptainCook4D, uses saved prediction artifacts for evaluation, and reports a clear F1 gain over the Count-Based baseline on the reproduced benchmark.

The problem-definition files are method-agnostic, and the method details are kept in `briefing/method.md` and `method/`. The evaluation code does not import from `method/`.

## Issues Found

- **Minor**: `reference.json` assigned a nonzero coefficient to `worst_quartile_f1`, but the main `scores.json` did not include that metric for the reproduced rows.
- **Minor**: `action.yaml` was still set to submit a follow-up CPU rerun. Final verification should leave the paper in a completed state.
- **Cosmetic**: `method/tgml.py` described a Count-Based baseline in the module docstring even though the baseline implementation lives outside that file.
- **Pre-existing workspace noise**: git status still shows harness/system artifacts (`README.md`, `manifest.json`, deleted `trajectory/*`). These were not created by this verification pass and are outside the gym code path.

## Fixes Applied

- Updated `eval/evaluator.py` so the main evaluator computes and writes `worst_quartile_f1`.
- Refreshed `scoring/scores.json` by rerunning standalone Count-Based and DO evaluation.
- Updated `scoring/TARGETS.md` to document `worst_quartile_f1`.
- Added `environment/runtime/` to `.gitignore` to keep generated container-runtime files out of future status.
- Corrected the stale `method/tgml.py` docstring.
- Updated `PROGRESS.md` to mention the complete scored metric set and to avoid describing a pending job.
- Set `action.yaml` to `done` at `core_claim_plus`.

## Final Judgment

This workspace is ready to serve as a research gym for the scoped CaptainCook4D task-graph generation claim. A scientist agent can use the method-agnostic problem/evaluation/data files in from-scratch mode, or use the full method implementation and scores in improve mode.

The main limitation is coverage: the gym does not reproduce EgoPER, EgoProceL, TGT text/video, or downstream task results. Within its declared CaptainCook4D scope, the scoring and evaluation pipeline are usable and discriminative.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: scoped to CaptainCook4D task-graph generation only
