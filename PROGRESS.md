# Progress

## What Works

- The released Direct Optimization (DO) / TGML method is present locally in `method/` and now matches the official preprocessing and masking more closely.
- CaptainCook4D loading works from the cloned repository assets, with 24 scenarios discovered.
- `eval/` remains method-agnostic and can score the count-based baseline or saved DO outputs.
- The patched code compiles with `python3 -m py_compile`.
- The scoring scope is now aligned to the implemented minimum experiment: CaptainCook4D task-graph generation only.
- `scripts/run.sh` now deletes stale scores before running the baseline and DO pipeline, then validates the fresh output.
- `environment/setup.sh` no longer depends on `pygraphviz`, reducing container build risk for the submitted job.
- The DO training loop no longer uses ground-truth graph F1 for checkpoint selection; it now follows the paper's weakly supervised sequence-accuracy stopping rule.
- The local count-based baseline now follows the released transition-graph logic instead of a simplified co-occurrence heuristic.
- A local CPU check on CaptainCook4D shows the corrected count-based baseline reaches about 67.1 precision, 56.8 recall, and 61.4 F1, which is close to the paper's 66.0 / 55.4 / 60.2.

## Results

- Current trustworthy milestone is still `method_runs`.
- Previously saved CaptainCook4D numbers in `scoring/scores.json` are not treated as valid for the corrected reproduction because the old training loop used ground-truth graph F1 for model selection.
- Paper reference target for the core claim: CaptainCook4D DO F1 = 87.8, compared against Count-Based F1 = 60.2.
- Local pre-submission sanity check for the corrected baseline: CaptainCook4D Count-Based F1 ≈ 61.4 from direct execution on the bundled annotations and graphs.

## Remaining

- Re-run CaptainCook4D end-to-end with the corrected DO implementation.
- Regenerate `checkpoints/do_captaincook4d/*.json` and `scoring/scores.json` from the corrected pipeline.
- Compare the corrected DO result against the count-based baseline and determine whether the core claim is supported.
- If the CaptainCook4D result supports the claim, update milestone and wrap-up docs; otherwise iterate on any remaining implementation issues.
- The next submission is configured to run the corrected count-based baseline and corrected DO method in one clean job, with stale `scores.json` and old DO outputs removed before execution.

## Issues

- The prior workspace state had two correctness problems:
- CaptainCook4D sequences were missing the explicit `START` token required by the official DO code.
- `scripts/evaluate.sh` and `eval/evaluator.py` could fall back to the count-based baseline while labeling the result as `do`.
- `method/train.py` used ground-truth graph F1 to choose the best checkpoint, which leaks supervision and invalidates any milestone above `method_runs` from that run.
- The previous local count-based implementation was too simplified relative to the paper's released transition-graph baseline, making the baseline comparison less trustworthy before this turn's fix.
- Some workspace files such as `data/loader.py` live under ignored paths, so the compute run depends on the working tree rather than only committed files.

## Deviations from Paper

- No algorithm substitution has been made.
- The current run plan focuses on CaptainCook4D first, which is the paper's primary task-graph-generation benchmark and the minimum setting needed for the core claim.
- `scoring/reference.json` is intentionally scoped to CaptainCook4D for this submission because EgoPER and EgoProceL loaders are not yet packaged in the current workspace. This narrows the benchmark to the implemented experiment rather than claiming unsupported coverage.
