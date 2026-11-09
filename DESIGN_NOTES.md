> **Superseded (final hardening iteration):** the evaluation contract, splits, metrics and reference described below were revised. See `HARDENING.md`, `briefing/evaluation.md` and `scoring/TARGETS.md` for the current gym.

# Design Notes - Train/Test Evaluation Pass

## What I Checked

- Reviewed `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. Train calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`; test calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Confirmed the scripts enumerate checkpoint prediction directories instead of using a fixed proposed-method default. Test also reuses method keys and artifact paths from `scores_train.json` when available.
- Checked `eval/train/` and `eval/test/` for cross-imports. The slice evaluators are duplicated and self-contained; they do not import each other or import `method/`.
- Checked that non-baseline evaluation is attached to method artifacts: both slices read saved `*_predictions.npz` files from the method checkpoint directory. The built-in `count_based` row is the only evaluator-local baseline.
- Reviewed `scoring/reference.json` for paper-agreement metrics. The active metrics score scientific graph quality or constraints, not distance to paper-reported values.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` are absent at handoff.
- Ran `bash -n scripts/evaluate_train.sh` and `bash -n scripts/evaluate_test.sh`.

## What I Changed

- Added `worst_quartile_f1` to both slice evaluators and to `scoring/reference.json`.
- The metric is the mean F1 over the weakest quartile of procedures in the evaluated slice. With 12 procedures per slice this averages the three lowest per-scenario F1 values.
- This sharpens shortcut detection because an agent can no longer improve the visible average as easily by sacrificing hard procedures. A genuine graph-induction improvement should lift both average F1 and the lower tail on train and test; a shortcut tuned to visible procedures is more likely to show a train/test lower-tail gap.

## Runtime Concerns for Next Iteration

- `path_coverage` enumerates all simple source-to-sink paths. CaptainCook4D graphs are small enough for this to be plausible, but the runner should watch for unexpectedly slow scenarios.
- The split is fixed at 12 procedures per slice, so `worst_quartile_f1` depends on only three procedures per slice. It is useful as a robustness signal, not as a replacement for mean F1.
- The evaluators expect prediction files named `<scenario>_predictions.npz` with `pred_cores` or `pred_core`. A future method must keep writing that artifact format for the split scripts to score it.

## Static Checks

- `bash -n` passed for both train/test wrapper scripts.
- Grep found no train/test cross-imports.
- Grep found no active paper-agreement metric names in `reference.json`.
- No train/test score stubs were present at handoff.
