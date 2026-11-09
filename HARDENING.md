# Hardening report

**Verdict.** Better results now score higher and worse results lower, by margins well above rerun noise.

- **Reference.** It now implements the paper's full Direct Optimization protocol, including the Sequence-Accuracy early stopping the earlier package omitted. It scores **88.35 F1** on the held-out CaptainCook4D procedures (paper: 87.8).
- **Shortcuts.** Every major shortcut in `REDTEAM.md` is closed: it now fails or ties the reference only by submitting the reference's own predictions.
- **Headroom.** On CaptainCook4D alone it is thin, so I added a weighted second experiment where the paper's method falls short: demonstrations perturbed by the paper's own insert/delete/replace noise model, with hidden graphs. Principled variants beat the reference there by 9–13 F1.
- **Not closable.** CaptainCook4D's held-out graphs are also public upstream on GitHub. A method that re-downloads them still pays on the 0.7-weight experiment; the grader has to catch that (see "Shortcuts left").

The evidence lives in `notes/hardening/`:
- `run_hardening.sh` replays every case through the stock no-argument `scripts/evaluate_test.sh` / `evaluate_train.sh`;
- `results/summary.md`, `results/analysis.json` and `results/test_combined.json` hold the after-numbers;
- `job1/` holds the first audit, made before the second experiment existed.

Rank deltas (Δrank) follow the ranking rule: |coefficient| × improvement in the better direction ÷ the metric's spread across the audit population, summed over metrics and weighted by experiment. Populations differ between the redteam audit (one experiment) and this one (two experiments). Compare signs and orderings, not absolute sizes.

## 1. Does a better result score higher?

**Before (`REDTEAM.md`, CaptainCook4D held-out, reference without early stopping):**

| Variant | Test F1 | Δrank |
|---|---:|---:|
| Reference (saved) | 76.19 | 0 |
| "Better": checkpoint chosen by Sequence Accuracy | 87.17 | +0.737 |
| 3,000 steps | 75.10 | −0.037 |
| One step | 10.92 | −4.928 |
| Count-Based | 61.40 | −1.026 |
| Bounded enumeration (brute force) | 30.18 | −3.177 |

The redteam's "better" variant is essentially the paper's own early stopping, so it is now part of the reference.

**After.** All variants were trained from the final code on every procedure of both experiments and scored by the hidden test. Columns: F1 on held-out procedures; ΔF1 with paired 95% bootstrap CI over procedures; weighted Δrank.

| Variant | CaptainCook4D F1 (ΔF1, CI) | Noisy F1 (ΔF1, CI) | Combined Δrank |
|---|---|---|---:|
| **Reference** `do`: DO + TGML + SA early stopping | 88.35 | 68.05 | 0 |
| β chosen per experiment on the public procedures (0.005 for CaptainCook4D, 0.5 for noisy) | 88.35 (±0) | **81.16** (+13.11, [+10.7, +15.6]; better on 80/100, worse on 18) | **+0.453** |
| Always-before heuristic (transitive reduction of orders seen in every demonstration) | **89.58** (+1.22, [−0.22, +3.89]) | 77.48 (+9.43, [+7.3, +11.6]) | **+0.404** |
| Majority-vote heuristic (order in ≥ 80% of demonstrations) | 82.15 (−6.20) | 78.53 (+10.48, [+7.6, +13.4]) | +0.026 |
| β chosen per seed by Sequence Accuracy (paper's Ego-Exo4D protocol) | 88.35 (±0) | 68.32 (+0.27, [−0.44, +0.83]) | +0.013 |
| Checkpoint chosen by Sequence Accuracy | 88.21 (−0.14) | 67.90 (−0.15) | +0.001 |
| Seed ensemble (mean Z of 5 / 20 seeds) | 88.19 / 88.19 | 67.98 / 68.14 | −0.017 / −0.011 |
| Fixed β = 0.05 | 84.81 (−3.54) | 69.11 (+1.07) | −0.170 |
| Fixed β = 0.5 | 73.45 (−14.90) | 81.16 (+13.11) | −0.426 |
| No early stopping (the previous reference) | 76.19 (−12.16) | 68.62 (+0.57) | −0.639 |
| Count-Based baseline (built into the evaluator) | 61.40 (−26.95, [−32.3, −21.3]) | 63.36 (−4.69, [−8.41, −0.93]) | −1.669 |
| One Adam step | 9.40 | 9.88 | −5.489 |

**Rerun noise.** The reference protocol was rerun with three other five-seed sets:
- largest |ΔF1|: 0.56 on CaptainCook4D and 0.09 on noisy;
- largest |combined Δrank|: 0.030.

Better variants exceed this by more than 10×, and their paired CIs on the noisy experiment exclude zero. Seeds are fixed, so the reference itself reproduces bit-for-bit across jobs: 24/24 CaptainCook4D files identical between jobs, and all 224 procedures identical on re-derivation.

**Train/test agreement** (public → held-out F1): honest methods score alike, while a method fitted to public labels collapses on test.

| Method | CaptainCook4D | Noisy |
|---|---|---|
| Reference | 88.55 → 88.35 | 66.98 → 68.05 |
| Always-before | 89.74 → 89.58 | 76.14 → 77.48 |
| β = 0.5 | 69.99 → 73.45 | 79.75 → 81.16 |
| Count-Based | 61.43 → 61.40 | 62.65 → 63.36 |
| Copying public labels | 99.79 → 17.24 | 100.0 → 21.29 |

## 2. Major shortcuts: fix, before, after

| Shortcut | Fix | Before (CaptainCook4D F1, Δrank) | After (CaptainCook4D / noisy F1, combined Δrank) |
|---|---|---|---|
| Copy public ground-truth graphs (`visible_labels`) | Held-out graphs live only in `eval/test/`. `download.sh` fetches a pinned tarball with no git history and strips held-out edges, keeping the node list. Held-out copies were deleted from `method/repo/data/` and `redteam/`. Noisy held-out graphs were never public. | 100.0, **+2.007** | 17.24 / 21.29, −4.986 (nothing to copy) |
| Rewrite the public ground-truth files to match a trivial output | The hidden test reads ground truth only from `eval/test/`; public files are never used for test scoring. | 100.0, **+2.007** | 17.24 / 21.29, −4.986 |
| Score a partial subset via the registry key `captaincook4d_partial_compare` | Partial-compare dispatch removed. Methods are discovered from `checkpoints/*/`, never from `scores_train.json`. Every procedure is required, otherwise the method fails at the floor. | 80.0 (1 of 12 procedures), **+0.521** | failed (0 / 0), −8.322 |
| Declare all seed ids = 0 for a free `std_f1 = 0` | `std_f1` is no longer scored (the evaluator cannot verify seeds); trial spread is an unscored diagnostic over trial index. | 76.19, **+0.059** | 88.35 / 68.05, 0.000 (identity: these are the reference's own predictions) |
| Submit one seed | Exactly 5 trials per procedure required. | 76.37, **+0.069** | failed, −8.322 |
| Submit one seed copied into 5 trials | (same rule) | n/a | 88.19 / 68.08, −0.011 (noise; no `std_f1` to gain) |
| Empty / NaN / ±inf / out-of-range / wrong-shape artifacts | Shape (5, n, n), finite values in [0, 1] required, else the run fails at the floor of every metric. | NaN scores written with exit 0 (empty); NaN/inf scored like all ones | all failed, −8.322 |
| Forged score files (F1 = 1e9, `success=true`) | The test entry point deletes and rebuilds `scores_test.json`; only `pred_cores` is read. | 17.24 (already failed) | 17.24 / 21.29, −4.986 |
| Replace `baseline/count_based.py` | The baseline lives inside each evaluator. | identity tie | unchanged (evaluator-owned baseline) |
| Stock runtime failure (`ModuleNotFoundError: networkx.algorithms.core`) | The broken vendored `environment/pylibs` was removed from `PYTHONPATH` (`scripts/common_env.sh`, `environment/container.def`) and deleted. | test exited 1 before scoring | runs in a fresh container (networkx 3.4.2 from the image) |

The redteam's brute-force run (bounded enumeration under the TGML objective, 30.18 F1) was not replayed. Its code was removed with `redteam/` because that directory held label-derived artifacts. None of the fixes makes enumeration easier, and the noisy experiment's graphs (9–27 nodes) are equally out of reach.

## 3. Shortcuts left, and why

1. **Re-downloading CaptainCook4D's graphs from the upstream repository.** CaptainCook4D is a public dataset: the 12 held-out graphs exist at the pinned GitHub commit. A method that fetches them scores ~100 F1 on the 0.7-weight experiment. A public-data gym cannot remove this. `scoring/CONSTRAINTS.md` forbids it explicitly, and the grader has to check for it. The noisy-demonstration experiment cannot be gamed this way: its held-out graphs and generator seed exist only in `eval/test/`.
2. **Qualitative graph figures in `paper/`** (images of DO outputs and some ground-truth graphs). Reading edges off images is laborious and partial. Left as is.
3. **Using public graphs as training labels** (for example, learning a pairwise edge classifier on the 112 public procedures). This is a constraint violation the evaluator cannot see. It is not an answer leak: the held-out graphs differ.
4. **Trials are not provably independent seeds.** A deterministic or duplicated submission is accepted. Since seed spread is no longer scored, this gains nothing beyond noise: −0.011 measured.

## 4. Reference score before and after

| | Before | After | Paper (24 procedures) |
|---|---|---|---|
| CaptainCook4D held-out P / R / F1 | 77.02 / 75.41 / 76.19 | **87.26 / 89.85 / 88.35** | 86.4 / 89.7 / 87.8 |
| CaptainCook4D public F1 | 76.53 | 88.55 | — |
| Noisy demonstrations held-out / public F1 | — | 68.05 / 66.98 | — |

What changed in the reference:
- **Training procedure only.** The paper's Sequence-Accuracy early stopping (Appendix A-B/A-C, released `train_with_gt.py`) is now applied, and the state at the halting epoch is the output. Loss, model, optimizer and post-processing are unchanged.
- **β rule.** The paper's per-procedure β rule (0.5 when no demonstration contains every step) never fires on CaptainCook4D, and fires on 1 of 100 procedures per noisy split.
- **Faster loss and SA, same results.** Training uses an exact batched form of the TGML loss, checked against the scalar loss at every run start. It also uses a vectorised Sequence Accuracy, cross-checked against the released implementation at 7 epochs per run; it agreed in all 1,120 reference runs of the final job (5 seeds × 224 procedures).

The paper's claim keeps its direction. On CaptainCook4D, DO beats Count-Based by 26.95 F1 on held-out procedures (better on 12/12, CI [+21.3, +32.3]). On noisy demonstrations it wins by 4.69 F1 (CI [+0.93, +8.41]).

## 5. Headroom check

**No scored metric of the reference is at its bound.**

| Metric | CaptainCook4D reference (bound) | Noisy reference (bound) |
|---|---|---|
| precision | 87.26 (100) | 64.56 (100) |
| recall | 89.85 (100) | 72.23 (100) |
| f1 | 88.35 (100) | 68.05 (100) |
| worst_quartile_f1 | 76.46 (100) | 50.91 (100) |
| edge_density_ratio | 0.953 (1) | 0.885 (1) |
| path_coverage | 0.582 (1) | 0.147 (1) |
| topological_distance | 0.016 (0) | 0.112 (0) |

**CaptainCook4D alone has little headroom.** The best honest alternative found, the always-before heuristic, gains +1.22 F1 and +0.097 Δrank. That is above rerun noise (≤ 0.56 F1, ≤ 0.048 Δrank), but its graphs differ from the reference's on only 2 of 12 procedures, and the paired CI [−0.22, +3.89] includes zero. With clean, scripted demonstrations, any method consistent with every demonstration converges to nearly the same graph.

**What I added: `noisy_demonstrations_task_graph`, weight 0.3.**
- **Design.**
  - Graphs match CaptainCook4D in size and shape: 1.23 vs 1.27 edges per node, depth 0.65·n vs 0.64·n.
  - Each procedure has 4–10 random valid demonstrations, perturbed by the paper's Section IV-F3 model (replace / delete / insert with probability p).
  - 100 public and 100 held-out procedures.
  - The generator and held-out graphs are in `eval/test/build_noisy_benchmark.py` and `eval/test/noisy_demos_test.json`.
- **Choosing p.** A rule fixed before scoring, run on public procedures only (`notes/hardening/noise_level_selection_40.json`): take the hardest p in {0.20, 0.10, 0.05} at which the reference beats Count-Based by ≥ 5 F1.
  - At 0.20, the reference lost to Count-Based (48.5 vs 60.9).
  - At 0.10, it also lost (55.4 vs 63.6).
  - At 0.05, it won by 8.3, so 0.05 was chosen.
- **Enlargement.** The first draw had 40 procedures per side. There, Count-Based came within 2.55 F1 of the reference on held-out procedures, with a CI including zero. The benchmark was enlarged to 100 + 100 at the same p. Now the baseline is clearly below (−4.69, CI [−8.41, −0.93]) and the train/test values agree.
- **Why it has headroom.** The paper's method falls short here because its SA early stopping rarely triggers on noisy data (22–25% of runs vs 100% on CaptainCook4D), and its default β = 0.005 over-trusts noisy orders. Honest alternatives gain 9–13 F1, and the reference sits far from every bound.

**Brute force** cannot solve either experiment: 2^O(n²) candidate graphs for 9–27 nodes, and enumeration got 30 F1 on CaptainCook4D.

## 6. Weight and coefficient changes

- **`captaincook4d_task_graph` 1.0 → 0.7; new `noisy_demonstrations_task_graph` 0.3.** CaptainCook4D is the paper's main benchmark and keeps the majority. The noisy experiment covers the robustness the paper motivates (EgoProceL, the online-mistake-detection noise study), and it is where differences between methods are measurable.
- **`std_f1` (−0.15) removed.** Seed spread cannot be verified from submitted matrices, and self-declared seed ids were a shortcut. The paper's ±1.5 / ±8.0 are 90% CI half-widths, now kept under `_paper_ci90_half_width` in `reference.json`; the paper's reported P/R/F1 values are unchanged.
- **Coefficients are non-negative magnitudes.** Direction comes from `higher_is_better`; `topological_distance` changed from −0.18 to 0.18. The other coefficients are unchanged.

## 7. What could not be fixed (plainly)

- **Upstream re-download** (§3.1): it remains possible and pays on the CaptainCook4D experiment. Only the post-hoc grader can catch it.
- **Thin CaptainCook4D headroom:** it is a property of the dataset (12 held-out procedures, clean scripted demonstrations), not of the evaluator. Most of the measurable headroom is in the noisy experiment.
- **The noisy experiment is semi-synthetic.** Its graphs are calibrated to CaptainCook4D, but they are not real procedures, and the paper reports no task-graph numbers under noise.
- **The baseline margin on noisy demonstrations is modest** (4.7 F1), because Count-Based is noise-robust (its F1 barely changes from 0 to 20% noise) while the paper's method degrades.
- **Halves vs full benchmark:** paper numbers average 24 procedures; the gym scores two fixed 12-procedure halves. No setup was scaled down.
- **Per-procedure held-out F1 of audit variants in git history.** The harness auto-committed `notes/hardening/results/*details.json` from the audit jobs: per-procedure F1 of each audit variant on held-out procedures. These are scores, not graphs, and are removed from the final tree. History cannot be rewritten from this workspace. At most they would let someone pick, per held-out procedure, among the audit variants; that is selection on test, which the grader should treat as a shortcut.
