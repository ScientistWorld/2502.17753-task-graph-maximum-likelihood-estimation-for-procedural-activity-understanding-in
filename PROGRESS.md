# Progress

## What Works

- The released Direct Optimization (DO) / TGML method is present locally in `method/` and now matches the official preprocessing and masking more closely.
- CaptainCook4D loading works from the cloned repository assets, with 24 scenarios discovered.
- `eval/` remains method-agnostic and can score the count-based baseline or saved DO outputs.
- The patched code compiles with `python3 -m py_compile`.

## Results

- Current trustworthy milestone is still `method_runs`.
- Previously saved CaptainCook4D numbers in `scoring/scores.json` are not treated as valid for the corrected reproduction because they were produced before the preprocessing/evaluation fixes.
- Paper reference target for the core claim: CaptainCook4D DO F1 = 87.8, compared against Count-Based F1 = 60.2.

## Remaining

- Re-run CaptainCook4D end-to-end with the corrected DO implementation.
- Regenerate `checkpoints/do_captaincook4d/*.json` and `scoring/scores.json` from the corrected pipeline.
- Compare the corrected DO result against the count-based baseline and determine whether the core claim is supported.
- If the CaptainCook4D result supports the claim, update milestone and wrap-up docs; otherwise iterate on any remaining implementation issues.

## Issues

- The prior workspace state had two correctness problems:
- CaptainCook4D sequences were missing the explicit `START` token required by the official DO code.
- `scripts/evaluate.sh` and `eval/evaluator.py` could fall back to the count-based baseline while labeling the result as `do`.
- Some workspace files such as `data/loader.py` live under ignored paths, so the compute run depends on the working tree rather than only committed files.

## Deviations from Paper

- No algorithm substitution has been made.
- The current run plan focuses on CaptainCook4D first, which is the paper's primary task-graph-generation benchmark and the minimum setting needed for the core claim.
