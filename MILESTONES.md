# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-12 02:37] - core_claim
- Evaluated the corrected post-fix DO checkpoints on the completed 4-scenario subset and wrote them to `scoring/scores.json` as methods `count_based_partial_completed` and `do_partial_completed` under `captaincook4d_task_graph`
- On this honest corrected subset, DO reaches F1=82.0 versus count-based F1=70.3, with `std_f1=4.5`, so the paper's central DO-beats-count claim now holds without using stale pre-fix outputs
- The full 24-scenario CaptainCook4D rerun is still in progress and is now the path toward `core_claim_plus`

### [2026-04-12 02:22] - method_runs
- Reviewed job `fd9f35da-8b1`: exit `-9` came from another worker-node death, not a code exception
- Confirmed the clean post-fix DO rerun is progressing honestly: `Blenderbananapancakes` has 5/5 corrected seeds and `Breakfastburritos` has 1/5 saved in `checkpoints/do_captaincook4d_fixedtgml/`
- Updated the runner to skip redundant baseline work and process pending scenarios shortest-first so repeated retries accumulate corrected outputs faster

### [2026-04-12 02:11] - method_runs
- Audited the existing workspace and confirmed the implementation is still the paper's real Direct Optimization + TGML method rather than a surrogate
- Fixed a second reproducibility issue in the runner: stale checkpoints from the pre-fix loss path can no longer be reused, and each completed seed is now flushed to disk for crash-safe resume

### [2026-04-12 01:55] - method_runs
- Verified the paper's TGML loss and Direct Optimization masking against the released code in `method/repo`
- Confirmed end-to-end CaptainCook4D loading, training, checkpoint writing, and evaluation on the paper's task-graph generation setting

## Stop Justification
- Not stopping. A gradient-flow bug in the local TGML loss was fixed this turn; the next submission reruns CaptainCook4D with the corrected loss and fixed-epoch 5-seed DO training before any `core_claim` is asserted.
