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
- The current broad `Bootstrap: scratch` fallback has now been simplified so `container.def` is OS-only and `environment/setup.sh` handles Python package installation, matching the platform contract more closely.

## Results

- Current trustworthy milestone is still `method_runs`.
- Previously saved CaptainCook4D numbers in `scoring/scores.json` are not treated as valid for the corrected reproduction because the old training loop used ground-truth graph F1 for model selection.
- Paper reference target for the core claim: CaptainCook4D DO F1 = 87.8, compared against Count-Based F1 = 60.2.
- Local pre-submission sanity check for the corrected baseline: CaptainCook4D Count-Based F1 ≈ 61.4 from direct execution on the bundled annotations and graphs.
- `python validate.py --compare` now passes locally with fresh baseline scores written to `scoring/scores.json`.

## Remaining

- Re-run CaptainCook4D end-to-end with the corrected DO implementation.
- Regenerate `checkpoints/do_captaincook4d/*.json` and `scoring/scores.json` from the corrected pipeline.
- Compare the corrected DO result against the count-based baseline and determine whether the core claim is supported.
- If the CaptainCook4D result supports the claim, update milestone and wrap-up docs; otherwise iterate on any remaining implementation issues.
- The planned submission would have run the corrected count-based baseline and corrected DO method in one clean job, with stale `scores.json` and old DO outputs removed before execution.
- CaptainCook4D is still the only packaged experiment for this turn; downstream tasks and extra datasets remain out of scope for the pending submission.
- The environment build must be retried after switching `environment/container.def` from `Bootstrap: library` to `Bootstrap: docker`, because this cluster has no default Apptainer library remote configured.
- The Docker-backed retry then failed on Docker Hub anonymous pull limits, so the container base was moved again to a non-Docker-Hub public registry.
- The Quay retry also failed before build execution with malformed registry metadata, so the base image was moved again to Microsoft Container Registry (`mcr.microsoft.com`) for the next attempt.
- The MCR retry produced the same malformed OCI metadata error, so the recipe now avoids OCI registries entirely by switching to `Bootstrap: debootstrap` for Ubuntu Jammy.
- The debootstrap retry failed because `debootstrap` is not installed on the builder host, so the recipe now uses `Bootstrap: localimage` from the existing `/home/user/environment/base_container.sif` instead of any external bootstrap mechanism.
- The first localimage retry failed because the builder does not see the workspace at `/home/user` during build, so `From:` was changed to the workspace-relative path `environment/base_container.sif`.
- The relative localimage path was then resolved against the scheduler's infrastructure checkout rather than the paper workspace, so `From:` is now set to the host-side workspace path reported by the build logs.
- The host-path localimage retry progressed into unpacking the cached `.sif` but then failed while changing ownership inside the extracted rootfs (`set_attributes ... Invalid argument`), which points to a fakeroot/chown incompatibility rather than a missing file.
- The next retry replaces bootstrap unpacking entirely with a `Bootstrap: scratch` recipe that copies only the required host runtime pieces through `%files`, avoiding OCI pulls, `debootstrap`, and localimage ownership rewrites.
- The first scratch retry reached `%files` copying but failed because the builder host does not provide `/usr/local/bin/uv`, so the environment can not assume `uv` exists outside the runtime node.
- The second scratch retry then failed on `/usr/lib/python3.12`, confirming the builder host also does not match the login node's Python minor-version layout.
- The next retry keeps the broad scratch `%files` layout but removes all Python installation from `container.def`, leaving only directory copies plus environment setup.
- The subsequent retry narrows `%files` further to explicit binaries plus Python/library directories after the broad `/bin` copy failed on unreadable `sudo*` binaries during build.
- The next build error showed that the builder host does not expose `dash` reliably at `/bin/dash`, so the scratch recipe now sources `dash` from `/usr/bin/dash` and places it at both `/bin/dash` and `/bin/sh` inside the container.
- The following retry showed the builder host also could not stat `/usr/bin/dash`, so the shell fallback is now simplified further: `/bin/bash` is copied as both `/bin/bash` and `/bin/sh`, avoiding `dash` entirely.
- The newest retry then failed because the builder host could not stat `/usr/bin/python3.12`, so the scratch image now avoids version-pinned Python binaries and instead copies only `/usr/bin/python3` plus broad library directories (`/usr/lib`, `/usr/lib64`, `/usr/local/lib`) with `/lib` and `/lib64` symlinked in `%post`.
- That broader-library retry still failed because recursive copying of `/usr/lib` hit unreadable or dangling `.build-id` and GCC entries on the builder host, so the scratch image is now reduced again to the Python stdlib trees plus the exact shared libraries required by `/usr/bin/python3` and core modules like `_ssl`, `_bz2`, `_lzma`, `_sqlite3`, `readline`, and `_ctypes`.
- The next retry then failed because the builder host does not expose `/usr/lib/python3`, only the versioned stdlib path, so the container now copies only `/usr/lib/python3.12` and `environment/setup.sh` bootstraps `pip` via `python3 -m ensurepip --upgrade` instead of depending on distro-packaged `pip` files.
- The following retry then failed because the builder host also does not expose `/usr/lib/python3.12`, which confirms the login node and builder use different Python layouts entirely; the scratch recipe now vendors a full Python runtime bundle under `environment/runtime/` and copies it from the workspace instead of from host system paths.
- The next retry showed that `%files` relative paths are resolved from the scheduler infrastructure checkout rather than the paper workspace, so the vendored runtime source paths now use the host-side absolute workspace path from the build logs.
- Once the vendored runtime was copied successfully, the build failed on `python3.12/sitecustomize.py` because that stdlib entry is a symlink to `/etc/python3.12/sitecustomize.py`; the vendored runtime now includes `/etc/python3.12` as well so the symlink resolves inside the container.

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
- The system stopped the paper after 10 consecutive environment build failures, so no further compute-side iteration is possible in this workspace.
- A local login-node attempt to install the missing Python packages with `pip --break-system-packages` failed with `No space left on device`, so local package augmentation is not a reliable fallback.

## Deviations from Paper

- No algorithm substitution has been made.
- The current run plan focuses on CaptainCook4D first, which is the paper's primary task-graph-generation benchmark and the minimum setting needed for the core claim.
- `scoring/reference.json` is intentionally scoped to CaptainCook4D for this submission because EgoPER and EgoProceL loaders are not yet packaged in the current workspace. This narrows the benchmark to the implemented experiment rather than claiming unsupported coverage.
- This submission still targets the local DO implementation rather than the official training script because the official path needs additional packages and plotting dependencies; the immediate priority is one clean end-to-end CaptainCook4D run through the scheduler.
