# Progress

## What Works

- **Coverage audit now prints directly in the submission job**: `scripts/run.sh` now logs the complete and partial CaptainCook4D scenario checkpoint coverage before training starts, which makes progress across unstable worker nodes easier to inspect from job logs
- **Resumable training keeps banking progress across node deaths**: job `d2b999b2-cad` died at the worker level with exit `-9`, but two new DO seeds were flushed to disk before failure, bringing `Cheesepimiento` to 5/5 corrected seeds
- **Audit-restored `core_claim` after reset**: a fresh audit from the lower reset baseline confirms the workspace still implements the paper's actual TGML + Direct Optimization method, and the existing corrected-subset evidence is still valid
- **Audit-restored standalone evaluation**: `scripts/evaluate.sh` now runs again on the current workspace instead of failing on missing `numpy`, while still preserving the clean vendored `networkx` ahead of the broken mounted package tree
- **Audit re-restored `core_claim`**: the current handed-off workspace still implements the paper's actual TGML + Direct Optimization method, and standalone partial evaluation continues to show the corrected completed subset supports the central DO-beats-count claim honestly
- **Audit restored `core_claim`**: the workspace still implements the paper's actual TGML + Direct Optimization method rather than a surrogate, and the corrected completed subset continues to support the central DO-beats-count claim honestly
- **Paper's algorithm fully implemented**: TGML loss (maximizing P of observed steps under DAG constraints), DirectOptimization model, and post-processing pipeline (binarize at 1/(n+2), boundary edges, transitive reduction)
- **CaptainCook4D loading**: 24 scenarios from cloned CaptainCook4D repo at `/home/user/method/repo/data/captaincook4d`
- **Evaluation pipeline**: Method-agnostic eval/ module scores edge-level P/R/F1. No imports from method/
- **Count-based baseline**: Matches paper's released code — F1=61.4 vs paper 60.2 ✓
- **DO training path repaired**: TGML loss now preserves autograd through the adjacency matrix, and a minimal smoke test confirms nonzero gradients and parameter updates
- **Training-side post-processing aligned more closely with the release**: `method/train.py` no longer applies cycle deletion inside the sequence-accuracy diagnostic path, matching the released `train_with_gt.py` behavior more closely
- **Environment packaging partially repaired**: `environment/setup.sh` installs a clean system `networkx`, while the shell entrypoints prepend `/home/user/environment/pylibs` to avoid the broken mounted user-site `networkx`; current jobs still rely on the mounted `torch` and `numpy` packages under `/home/user/.local`
- **Shell entrypoints hardened**: `scripts/run.sh`, `scripts/method.sh`, `scripts/evaluate.sh`, `scripts/baseline.sh`, and `scripts/reproduce.sh` now prepend `/home/user/environment/pylibs`, which contains a clean vendored `networkx==3.4.2`, so login-node evaluation no longer imports the corrupted user-site `networkx`
- **Resumable orchestration repaired**: `scripts/method.sh` now performs full-dataset DO evaluation only after all 24 CaptainCook4D scenarios have all requested seeds, so partial retries can bank corrected checkpoints without failing at the evaluation stage
- **Container**: `python:3.12-slim-bookworm` + libgomp1
- **GitHub repo cloned**: `/home/user/method/repo` contains the official TGML codebase

## Results

### captaincook4d_task_graph

| Method | Precision | Recall | F1 | std_f1 |
|--------|-----------|--------|-----|--------|
| Paper: count_based | 66.0 | 55.4 | 60.2 | — |
| **Reproduced: count_based** | **67.1** | **56.8** | **61.4** | — |
| Paper: DO (5-seed avg) | 86.4 | 89.7 | 87.8 | 1.5 |

**Count-based baseline MATCHES paper within 1.2 F1 points.**

**Fresh post-fix DO scores are now available on a still larger completed corrected subset.**
Using only scenarios with all 5 corrected seeds saved in `checkpoints/do_captaincook4d_fixedtgml/`, the current partial experiment is:

| Experiment | Method | Precision | Recall | F1 | std_f1 |
|-----------|--------|-----------|--------|----|--------|
| `captaincook4d_task_graph` corrected completed subset (8 scenarios), stored as `count_based` in `scores.json` | count_based | 73.3 | 62.1 | 67.2 | — |
| `captaincook4d_task_graph` corrected completed subset (8 scenarios), stored as `do` in `scores.json` | DO | 79.4 | 76.7 | 78.0 | 7.4 |

The completed corrected subset is now:
`Blenderbananapancakes`, `Breakfastburritos`, `Buttercorncup`, `Cheesepimiento`, `Microwaveeggsandwich`, `Pinwheels`, `Spicedhotchocolate`, `Zoodles`.

The next partially completed scenario is now `Microwavefrenchtoast` with 2/5 corrected seeds saved, so the resumable shortest-first rerun should extend the subset further without revisiting completed work.

This subset is non-trivial and still supports the paper's core claim: corrected DO clearly outperforms the count-based baseline on edge-prediction F1 while using the real TGML/DO training procedure rather than stale pre-fix outputs.

**Full-dataset DO scores are still not claimable.**
Only 7/24 scenarios currently have all 5 corrected seeds. `Cheesepimiento` has been started and should be finished next, so the stale pre-fix full-dataset DO entry is still being removed from `scores.json` during each retry until the corrected rerun is complete.

## Issues Encountered

1. **GT adjacency matrix orientation bug (FIXED)**: Initially built gt_adj[dst, src] = 1 (adjacency form), but evaluation uses graph form adj[u,v] = 1 meaning edge u→v. Fixed to gt_adj[src, dst] = 1 matching official graph orientation.
2. **Graph building with transpose bug (FIXED)**: Used `nx.DiGraph(pred_binary.T)` which created edges in wrong direction. Fixed to `nx.DiGraph(pred_binary)` matching official.
3. **delete_cycles discrepancy (FIXED)**: Official train_with_gt.py does NOT call delete_cycles — only transitive reduction. Removed from post-processing.
4. **Count-based dedup bug (FIXED)**: My count-based baseline deduplicated sequences internally, but official TransitionGraph doesn't. Fixed to not dedupe.
5. **Count-based matrix token stripping (FIXED)**: Correct condition for GT node IDs is `1 <= node < n_gt` (excluding END at matrix index n_gt), not `1 <= node <= n_gt`.
6. **Compute node failures**: 10+ consecutive job submissions failed with exit -9 due to cluster node hardware failures. Resolved by switching to CPU-only training and resumable scripts.
7. **Training selection leakage risk (FIXED IN SCRIPTS)**: `scripts/method.sh` previously selected the best epoch using GT F1 during training. Updated it to save only the final fixed-epoch model, keeping training weakly supervised.
8. **Autograd break in TGML loss (FIXED)**: `method/tgml.py` detached the adjacency matrix to NumPy inside the loss, so `loss.backward()` could not update the graph parameters. Replaced this with a torch-only implementation and trimmed padded sequences at the first `gamma` token.
9. **Stale checkpoint reuse risk (FIXED IN SCRIPTS)**: the rerun path would have reused old per-scenario `.npz` files from the broken-loss run whenever seed IDs overlapped, mixing invalid and valid outputs. The method now writes to a fresh `checkpoints/do_captaincook4d_fixedtgml/` directory by default and flushes each completed seed to disk immediately for crash-safe resume.
10. **Repeated worker-node deaths (MITIGATED IN SCRIPTS)**: the latest job again failed with exit `-9` because the worker node died mid-run, not because of a Python error. The runner now skips recomputing the already-trusted baseline and trains pending DO scenarios in ascending estimated workload so each retry banks more corrected outputs before a preemption or node loss.
11. **Partial corrected-subset scoring added (FIXED IN EVAL/RUNNER)**: the workspace no longer has to wait for all 24 scenarios before reporting any honest corrected DO result. `scripts/evaluate.sh captaincook4d_partial_compare ...` now scores only scenarios with 5 corrected seeds and writes them into the standard `count_based` and `do` slots in `scoring/scores.json`; this is documented as a temporary subset-based deviation until the full 24-scenario corrected rerun completes.
12. **Completion-first retry scheduling added (FIXED IN RUNNER)**: repeated node failures were leaving some scenarios nearly finished but not completed. `scripts/method.sh` now prioritizes scenarios with the fewest missing seeds before using workload as a tie-breaker, and the retry batch is now capped at `METHOD_MAX_SCENARIOS=1` so unstable jobs are more likely to finish `Cheesepimiento` before opening more new scenarios.
13. **Broken mounted Python packages (FIXED FOR SHELL ENTRYPOINTS)**: local validation exposed that the mounted `networkx` installation under `/home/user/.local/lib/python3.12/site-packages` was corrupted. A clean `networkx==3.4.2` is now vendored in `/home/user/environment/pylibs`, and all shell entrypoints prepend that path so both local and job-side evaluation avoid the broken package tree while still using the existing mounted `torch` and `numpy`.
14. **Partial-run full-eval trap (FIXED IN RUNNER)**: `scripts/method.sh` previously forced a full-dataset DO evaluation after every retry, even when only one pending scenario was intentionally being trained. The script now skips that full evaluation until the 24-scenario checkpoint set is complete.
15. **Test-node instability remains the dominant blocker**: repeated submissions on test resources are being killed by `Worker node died while job was running` before a Python traceback appears. The next retry stays on a normal CPU queue and now lets `scripts/method.sh` automatically pick the next shortest pending scenario rather than staying pinned to `Cheesepimiento`, which is already complete.
16. **Audit-induced score-file mixing (FIXED)**: running standalone baseline mode can overwrite the `count_based` row in `scoring/scores.json` with the full-dataset baseline while the DO row still reflects the corrected subset. The score file has been rewritten from `captaincook4d_partial_compare` so the on-disk comparison is once again a like-for-like 7-scenario subset.
17. **Corrupted mounted torch on compute nodes (FIXED IN PACKAGED ENVIRONMENT)**: job `5eac6b2f-963` failed immediately because the container still searched `/home/user/.local/lib/python3.12/site-packages`, where `torch` is missing `torch.masked.maskedtensor.core`. The packaged fix now installs clean overlay copies of `numpy`, `networkx`, and `torch`, removes `/home/user/.local` from `PYTHONPATH`, and sources `scripts/common_env.sh` from every shell entrypoint so future jobs resolve imports from the overlay instead of the broken mount.
18. **Standalone evaluation path broke under strict import isolation (FIXED)**: `scripts/evaluate.sh` initially failed on the login node because `common_env.sh` hid the mounted user-site `numpy`, and the first fallback attempt incorrectly reintroduced the broken mounted `networkx`. The evaluation script now appends the local site-packages only after the clean vendored path, so standalone scoring works again without contaminating the `networkx` import order.
19. **Reset-baseline audit completed cleanly**: after the milestone reset to `method_runs`, a fresh audit reconfirmed that the implementation is still the paper's real method, `validate.py --compare` passes, `scripts/evaluate.sh` works standalone, and the current subset scores remain numerically plausible relative to the paper.

## What Remains

- Rerun CaptainCook4D DO from scratch with the corrected differentiable loss
- Expand the corrected 5-seed rerun beyond the current 7-scenario subset
- Expand beyond the now-completed `Cheesepimiento` scenario by finishing the next shortest pending CaptainCook4D procedure
- Replace the partial `core_claim` experiment with a full 24-scenario corrected DO result in `scores.json`
- Continue resumable shortest-first DO reruns until all 24 CaptainCook4D scenarios have 5 corrected seeds
- If the corrected rerun still trails badly, inspect per-scenario failures and only then make tightly scoped training/evaluation fixes

## Implementation Details

- `method/tgml.py`: TGML loss (Eq. 4-5) + DirectOptimization model (Eq. 6-8) + post-processing (Fig. 3)
- `method/train.py`: Training loop for fixed-epoch DO training
- `method/train_parallel.py`: CPU parallel training
- `eval/evaluator.py`: Edge-level P/R/F1 evaluation (method-agnostic)
- `eval/metrics.py`: Post-processing functions (binarize, boundary edges, transitive reduction)
- `baseline/count_based.py`: Transition-graph count-based baseline
- `data/loader.py`: CaptainCook4D data loader from cloned repo

## Deviations from Paper

- No algorithm substitutions
- Scoped to CaptainCook4D only (primary benchmark)
- Previous local DO outputs were generated before fixing the gradient bug and are therefore not accepted as reproduction evidence
- The next submission restores the paper's 5-seed / 1000-epoch regime on CaptainCook4D with the corrected differentiable loss
- Evaluation aggregates per-scenario metrics into the reported dataset means
- Count-based baseline reimplemented without pandas/pygraphviz
- Because of repeated worker-node deaths, the current `scores.json` still reflects the completed corrected subset (7 scenarios) rather than the full 24-scenario benchmark
