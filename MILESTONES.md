# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-08 22:55] - method_runs
- Reworked the environment back to the platform contract: `environment/container.def` is OS-only again and `environment/setup.sh` no longer tries to download Python packages
- Switched the planned reproduction job to CPU-only, which is sufficient for the scoped CaptainCook4D DO optimization and avoids wasting GPU budget on adjacency-matrix training
- Added runtime sanity checks to `scripts/run.sh` and made `scripts/method.sh` device-configurable instead of forcing CUDA
- Revalidated locally that `environment/setup.sh` succeeds and that the Count-Based baseline still writes a clean `scoring/scores.json` with about 61.4 mean F1 on CaptainCook4D
- Confirmed with a new local DO smoke test on the CaptainCook4D `Coffee` scenario that the corrected training path runs end-to-end on CPU for 5 epochs and improves sequence accuracy from 0.405 to 0.587
- Fixed `environment/container.def` again so the scratch `%files` entries point to the live builder-visible workspace runtime under `/home/user/scratch/AgentWorkspaces/workspaces/2502.17753/environment/runtime/...`
- Kept the milestone at `method_runs`; the corrected DO run still needs to finish on the scheduler before `core_claim` can be claimed

### [2026-04-08 23:05] - method_runs
- The next scheduler retry failed before `setup.sh` because `%files` tried to copy `environment/runtime/etc/ssl/certs`, and that vendored certificate tree contains dangling symlinks on the builder host
- Removed the `/etc/ssl/certs` copy from the scratch recipe, since the current scoped CPU submission does not download anything during build or runtime
- Kept the milestone at `method_runs`; this is still an environment-only correction before the pending CaptainCook4D DO submission

### [2026-04-08 23:10] - method_runs
- The following retry got through `%files` but failed when Apptainer tried to run `%post` under fakeroot inside the scratch image
- Removed `%post` entirely, since it was only creating directories and the current container contract does not need build-time shell execution
- Kept the milestone at `method_runs`; the correction is still environment-only and the DO experiment has not run yet

### [2026-04-08 23:15] - method_runs
- The next retry showed that the container build itself now succeeds, but the scheduler still fails in the separate `setup.sh` phase because the scratch image is too minimal for Apptainer's fakeroot helper
- Replaced the scratch recipe with a `Bootstrap: localimage` definition that reuses the existing `environment/base_container.sif`, while keeping `setup.sh` as a no-op
- Kept the milestone at `method_runs`; this is still an environment-only fix before the pending CaptainCook4D DO submission

### [2026-04-08 23:20] - method_runs
- The `localimage` retry failed while unpacking `base_container.sif`, with `set_attributes` ownership-change errors on the extracted rootfs
- Replaced the local SIF bootstrap with `Bootstrap: docker` from `python:3.12`, keeping the environment minimal and relying on the already-populated `/home/user/.local/lib/python3.12/site-packages` for paper dependencies
- Kept the milestone at `method_runs`; no new experiment result exists yet

### [2026-04-08 00:00] - method_runs
- Re-opened the workspace and revalidated locally that the packaged CaptainCook4D Count-Based baseline still reaches about 61.4 mean F1
- Simplified the current `Bootstrap: scratch` fallback so `environment/container.def` is OS-only and all Python package installation happens in `environment/setup.sh` with `--break-system-packages`
- The first scratch retry in this turn failed because copying the entire `/bin` directory hit unreadable `sudo` binaries on the host, so the next retry narrows `%files` to explicit shell/coreutils binaries plus Python/library directories
- The narrowed retry then failed because the builder host could not stat `/bin/dash`, so the next environment revision sources `dash` from `/usr/bin/dash` and uses it for both `/bin/dash` and `/bin/sh` in the scratch image
- That retry failed as well because the builder host could not stat `/usr/bin/dash` either, so the next environment revision avoids `dash` entirely and copies `/bin/bash` as both `/bin/bash` and `/bin/sh`
- The latest retry then failed because the builder host could not stat `/usr/bin/python3.12`, so the next environment revision removes version-pinned Python paths and copies only `/usr/bin/python3` plus broad library directories with `/lib` and `/lib64` symlinked in `%post`
- That broader-library retry then failed because recursive copying of `/usr/lib` hit unreadable or dangling `.build-id` and GCC entries on the builder host, so the next environment revision copies only the Python stdlib trees and the exact shared libraries needed by the interpreter and core modules
- The newest retry then failed because the builder host does not expose `/usr/lib/python3`, so the next environment revision drops that unversioned path and bootstraps `pip` from `ensurepip` inside `/usr/lib/python3.12`
- That retry then failed as well because the builder host also does not expose `/usr/lib/python3.12`, so the next environment revision vendors the Python runtime under `environment/runtime/` and copies it from the workspace instead of from host system paths
- That vendored-runtime retry then failed because `%files` resolved `environment/runtime/...` relative to the scheduler infrastructure checkout instead of the paper workspace, so the next environment revision uses the host-side absolute workspace path from the build logs for every vendored runtime source
- After the vendored runtime started copying, the next build failed because `sitecustomize.py` inside `python3.12` was still an absolute symlink to `/etc/python3.12/sitecustomize.py`, so the next environment revision replaces that symlink with a real file inside the vendored stdlib
- The scheduler then stopped the paper after 10 consecutive environment build failures, so no corrected CaptainCook4D DO run could be submitted
- Kept the milestone at `method_runs`; this turn prepares a fresh end-to-end baseline+DO submission, but no new DO result exists yet

### [2026-04-07 09:31] - method_runs
- Fixed a baseline-evaluation bug where `eval/evaluator.py` was post-processing the Count-Based adjacency a second time, collapsing the orientation and producing 0.0 F1
- Re-verified locally that the corrected Count-Based path scores about 67.1 precision, 56.8 recall, and 61.4 F1 on CaptainCook4D
- Tightened `TARGETS.md`, `CHECKLIST.md`, and briefing docs so they reflect the actual scoped CaptainCook4D experiment that the next submission will score
- Kept the milestone at `method_runs` because fresh DO results from the corrected pipeline are still pending the submitted job

### [2026-04-07 09:36] - method_runs
- The first submit attempt failed during container build because `environment/container.def` used `Bootstrap: library`
- Switched the container base to `Bootstrap: docker` with the same `ubuntu:22.04` image so Apptainer does not depend on an unavailable library remote
- Kept the milestone at `method_runs`; no experiment results changed, this is only an environment fix before resubmission

### [2026-04-07 09:38] - method_runs
- The second submit attempt failed while fetching `ubuntu:22.04` from Docker Hub due to unauthenticated pull-rate limits
- Switched the Docker bootstrap image to `quay.io/toolbx/ubuntu-toolbox:22.04` to avoid Docker Hub rate limiting while keeping an Ubuntu 22.04 base
- Kept the milestone at `method_runs`; this is still an environment-only fix before the actual DO rerun

### [2026-04-07 09:40] - method_runs
- The Quay-based retry failed before `%post` with malformed registry metadata (`invalid character 's' after top-level value`)
- Switched the Docker bootstrap image to `mcr.microsoft.com/devcontainers/base:jammy` for another public Ubuntu 22.04-compatible source
- Kept the milestone at `method_runs`; no experiment outputs changed

### [2026-04-07 09:42] - method_runs
- The MCR-based retry hit the same malformed OCI metadata failure, indicating a broader OCI-registry fetch problem on the builder
- Switched `environment/container.def` to `Bootstrap: debootstrap` with Ubuntu Jammy to bypass OCI registry pulls entirely
- Kept the milestone at `method_runs`; still an environment-only correction before the actual experiment run

### [2026-04-07 09:45] - method_runs
- The debootstrap-based retry failed because `debootstrap` is not available on the builder host
- Repointed `environment/container.def` to `Bootstrap: localimage` using the existing `/home/user/environment/base_container.sif` so the next build can clone a local base image without network/bootstrap dependencies
- Kept the milestone at `method_runs`; no experimental results changed

### [2026-04-07 09:47] - method_runs
- The first localimage retry failed because the builder could not resolve `/home/user/environment/base_container.sif`
- Changed the localimage source to the workspace-relative path `environment/base_container.sif` so it resolves inside the build context instead of assuming the runtime mount point
- Kept the milestone at `method_runs`; still an environment-only fix

### [2026-04-07 09:49] - method_runs
- The relative localimage retry failed because the builder resolved `environment/base_container.sif` against its infrastructure checkout instead of the paper workspace
- Updated `environment/container.def` to use the host-side absolute workspace path from the scheduler log for `base_container.sif`
- Kept the milestone at `method_runs`; no experiment results changed

### [2026-04-07 10:02] - method_runs
- The host-path localimage retry reached rootfs extraction but failed on `set_attributes` ownership changes inside the unpacked image, so the remaining issue is builder-side chown behavior rather than missing bootstrap inputs
- Replaced the bootstrap strategy with a `Bootstrap: scratch` recipe that copies the local Python/bash runtime via `%files`, avoiding external registries, host `debootstrap`, and localimage unpack ownership rewrites
- Kept the milestone at `method_runs`; this is still an environment-only correction before the CaptainCook4D DO rerun

### [2026-04-07 10:08] - method_runs
- The first scratch retry failed while copying `/usr/local/bin/uv`, showing that the builder host does not have the same `uv` path as the login node
- Removed `uv` from the scratch image recipe and switched `environment/setup.sh` to `python3 -m pip`, which is available from the copied system Python packages
- Kept the milestone at `method_runs`; no experiment outputs changed

### [2026-04-07 10:12] - method_runs
- The next scratch retry failed on the login-node-specific path `/usr/lib/python3.12`, so the builder host also differs in Python minor-version layout
- Replaced the version-pinned `%files` entries with broad system directories (`/bin`, `/usr/bin`, `/usr/lib`, `/usr/local/lib`, `/lib`, `/lib64`) to make the scratch recipe builder-agnostic
- Kept the milestone at `method_runs`; still an environment-only correction before the actual CaptainCook4D run

### [2026-04-07 10:15] - method_runs
- The system stopped the paper after 10 consecutive container or setup build failures, before any corrected CaptainCook4D DO job could be submitted
- Highest honest milestone remains `method_runs`: the paper's algorithm was audited and corrected locally, and the count-based baseline was validated locally, but the compute-side reproduction never ran
- Marked the workspace final with `action: done` because no further submissions will be accepted

### [2026-04-07 00:00] - method_runs
- Replaced the simplified local count-based baseline with a transition-graph implementation that matches the paper's released baseline logic closely
- Verified on CaptainCook4D annotations that the corrected baseline reaches about 61.4 mean F1 locally, which is on the same scale as the paper's reported 60.2 baseline
- Kept the milestone at `method_runs` because the proposed DO result still needs a fresh run after the baseline-fidelity correction

### [2026-04-05 04:45] - method_runs
- Audited the inherited workspace and confirmed the method is DO/TGML rather than a surrogate
- Found a training-time supervision leak: `method/train.py` was selecting the best epoch using ground-truth graph F1, which is not allowed in the paper's weakly supervised setup
- Replaced GT-based model selection with the official sequence-accuracy early-stopping criterion so the next run can support an honest milestone claim

### [2026-04-05 04:20] - method_runs
- Scoped the benchmark and scoring files to the implemented CaptainCook4D experiment so reference and scores can be compared honestly
- Updated the run script to delete stale scores, rerun the baseline and DO pipeline, and validate the fresh outputs
- Simplified `environment/setup.sh` to avoid a likely `pygraphviz` build failure on compute

### [2026-04-05 03:55] - method_runs
- Audited the local DO/TGML pipeline against the released repository bundled in `method/repo`
- Confirmed the implementation is the paper's algorithm, but found fidelity bugs that block a justified `core_claim`
- Fixed CaptainCook4D preprocessing to prepend `START`, aligning training sequences with the official DO code
- Fixed the DO adjacency mask to match the released implementation and aligned boundary-edge post-processing with the paper
- Removed the invalid evaluation fallback that silently scored the count-based baseline as `do`
- Recompiled the patched Python modules and verified the loader now emits `[beta, START, ... , END, gamma]` sequences on CaptainCook4D

## Stop Justification
- The system stopped this paper after 10 consecutive container or setup build failures.
- No compute job with the corrected DO implementation was ever submitted successfully, so the highest honest milestone remains `method_runs`.
