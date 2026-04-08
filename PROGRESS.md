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
- The evaluator now scores the Count-Based baseline correctly instead of post-processing its already-final adjacency a second time.
- The current `Bootstrap: scratch` fallback is now OS-only again, and the mounted workspace user site under `/home/user/.local/lib/python3.12/site-packages` provides the Python packages needed by the scoped CaptainCook4D run.
- `environment/setup.sh` now acts as a pure sanity check rather than trying to download packages at build time.
- `scripts/method.sh` is now device-configurable and `scripts/run.sh` defaults the planned submission to CPU execution.
- A new local DO smoke test on the CaptainCook4D `Coffee` scenario completed successfully for 5 CPU epochs, confirming that the corrected training path still executes after the recent environment changes.
- `environment/container.def` now references the builder-visible host workspace path `/home/user/scratch/AgentWorkspaces/workspaces/2502.17753/environment/runtime/...` for the vendored scratch runtime instead of the older stale path from previous scheduler attempts.

## Results

- Current trustworthy milestone is still `method_runs`.
- Previously saved CaptainCook4D numbers in `scoring/scores.json` are not treated as valid for the corrected reproduction because the old training loop used ground-truth graph F1 for model selection.
- Paper reference target for the core claim: CaptainCook4D DO F1 = 87.8, compared against Count-Based F1 = 60.2.
- Local pre-submission sanity check for the corrected baseline: CaptainCook4D Count-Based F1 ≈ 61.4 from direct execution on the bundled annotations and graphs.
- Local pre-submission sanity check for the corrected DO path: CaptainCook4D `Coffee` scenario completed for 5 CPU epochs with sequence accuracy improving from 0.405 to 0.587.
- `python validate.py --compare` now passes locally with fresh baseline scores written to `scoring/scores.json`.

## Remaining

- Re-run CaptainCook4D end-to-end with the corrected DO implementation.
- Regenerate `checkpoints/do_captaincook4d/*.json` and `scoring/scores.json` from the corrected pipeline.
- Compare the corrected DO result against the count-based baseline and determine whether the core claim is supported.
- If the CaptainCook4D result supports the claim, update milestone and wrap-up docs; otherwise iterate on any remaining implementation issues.
- The next planned submission runs the corrected count-based baseline and corrected DO method in one clean CPU job, with stale `scores.json` and old DO outputs removed before execution.
- CaptainCook4D is still the only packaged experiment for this turn; downstream tasks and extra datasets remain out of scope for the pending submission.
- The current submission therefore retries the scratch recipe with a smaller risk surface: no pip work in `container.def`, no dependency downloads in `setup.sh`, and an explicit dependency on the mounted workspace user site for Python packages.
- The latest container-build failure came from recursively copying `environment/runtime/etc/ssl/certs`, whose host-side certificate tree contains dangling symlinks; the next retry removes that `%files` entry entirely.
- The subsequent `localimage` retry failed during rootfs ownership rewriting while unpacking `base_container.sif`, so the next retry switches to a plain `python:3.12` Docker bootstrap instead of reusing any local SIF.
- The mounted workspace already contains the needed Python packages in `/home/user/.local/lib/python3.12/site-packages`, so a stock Python 3.12 base should be enough for the scoped CPU run.
- The first `python:3.12` retry failed before build-time execution with malformed OCI registry metadata (`invalid character 's' after top-level value`), which points to an external registry/network response rather than a local recipe problem.
- After the registry path failed twice with the same malformed-metadata response, the environment was moved back to the local scratch recipe and the vendored runtime was expanded with additional shared libraries (`libcap`, `libaudit`, `libseccomp`, `libpam`, `libcrypt`, `libresolv`, `libnsl`) aimed at satisfying Apptainer's fakeroot helper during `setup.sh`.

## Issues

- The prior workspace state had two correctness problems:
- CaptainCook4D sequences were missing the explicit `START` token required by the official DO code.
- `scripts/evaluate.sh` and `eval/evaluator.py` could fall back to the count-based baseline while labeling the result as `do`.
- `method/train.py` used ground-truth graph F1 to choose the best checkpoint, which leaks supervision and invalidates any milestone above `method_runs` from that run.
- The previous local count-based implementation was too simplified relative to the paper's released transition-graph baseline, making the baseline comparison less trustworthy before this turn's fix.
- `eval/evaluator.py` had been post-processing the Count-Based baseline output a second time, which inverted the useful prediction structure and caused an invalid 0.0-F1 result until corrected.
- Some workspace files such as `data/loader.py` live under ignored paths, so the compute run depends on the working tree rather than only committed files.
- The local login-node Python environment does not currently include all package dependencies, so full end-to-end verification must come from the submitted containerized job rather than this shell alone.
- Repeated Apptainer builder failures have all happened before `setup.sh` runs, so the current blocker is the base container recipe rather than Python package installation.
- The builder host and the login node do not expose the exact same tool paths, so `%files` entries must only reference binaries guaranteed to exist on the builder itself.
- The scratch recipe therefore has to avoid login-node-specific paths like `/usr/bin/python3.12` and `/usr/lib/python3.12`, relying instead on `/usr/bin/python3` and broad, version-agnostic library directories.
- The scratch recipe also cannot recursively copy host-maintained library trees like `/usr/lib`, because the builder hits unreadable or stale entries there; `%files` has to stay on explicit file paths and narrow Python directories.
- The builder host also does not guarantee an unversioned `/usr/lib/python3` tree, so the scratch recipe has to rely on the versioned stdlib path plus `ensurepip` rather than distro-specific `pip` package locations.
- More broadly, the builder host does not guarantee any of this login node's Python filesystem layout, so the remaining robust option is to vendor the needed runtime files inside the workspace and copy from there.
- The builder also does not resolve paper-relative `%files` sources reliably, so vendored workspace assets have to be referenced by their host-side absolute workspace paths.
- A local login-node attempt to install the missing Python packages with `pip --break-system-packages` failed with `No space left on device`, so local package augmentation is not a reliable fallback.
- The main remaining risk is still container build compatibility with the scheduler's builder host; the current submission is designed to test the simplified scratch recipe directly instead of assuming the previous stop condition is permanent.
- The most likely remaining failure mode is that the scheduler builder may still not resolve `/home/user/...` in `%files`; the current submission is explicitly testing that narrower path change.

## Deviations from Paper

- No algorithm substitution has been made.
- The current run plan focuses on CaptainCook4D first, which is the paper's primary task-graph-generation benchmark and the minimum setting needed for the core claim.
- `scoring/reference.json` is intentionally scoped to CaptainCook4D for this submission because EgoPER and EgoProceL loaders are not yet packaged in the current workspace. This narrows the benchmark to the implemented experiment rather than claiming unsupported coverage.
- This submission still targets the local DO implementation rather than the official training script because the official path needs additional packages and plotting dependencies; the immediate priority is one clean end-to-end CaptainCook4D run through the scheduler.
