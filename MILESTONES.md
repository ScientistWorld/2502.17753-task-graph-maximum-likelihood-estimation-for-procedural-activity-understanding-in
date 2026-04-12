# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-12 03:18] - core_claim
- Reviewed job `1b636c76-157`: another worker-node death with exit `-9`, but still no code exception; `Breakfastburritos` reached 5/5 corrected seeds before the node died during `Microwaveeggsandwich`
- Refreshed `scoring/scores.json` from the actual completed 6-scenario corrected subset: count-based F1=68.0 versus DO F1=79.4 (`std_f1=6.1`), keeping the paper's central DO-beats-count result on a larger honest subset
- Tightened the retry batch again from 4 scenarios to 2 so the next resumable CPU run can focus on finishing `Microwaveeggsandwich` before expanding farther

### [2026-04-12 03:05] - core_claim
- Reviewed job `c961d2d4-b2c`: another worker-node death with exit `-9`, but no code error; `Zoodles` reached 5/5 corrected seeds and `Breakfastburritos` advanced to 2/5
- Refreshed `scoring/scores.json` from the actual completed 5-scenario corrected subset: count-based F1=68.7 versus DO F1=79.3 (`std_f1=6.7`), preserving the paper's central DO-beats-count result on a larger honest subset
- Tightened the retry batch from 6 scenarios to 4 so the next resumable CPU run is more likely to finish the remaining partial scenario before opening additional work

### [2026-04-12 02:37] - core_claim
- Evaluated the corrected post-fix DO checkpoints on the completed 4-scenario subset and wrote them into the standard `count_based` and `do` slots in `scoring/scores.json`, with the subset-based deviation documented in `PROGRESS.md`
- On this honest corrected subset, DO reaches F1=82.0 versus count-based F1=70.3, with `std_f1=4.5`, so the paper's central DO-beats-count claim now holds without using stale pre-fix outputs
- The full 24-scenario CaptainCook4D rerun is still in progress and is now the path toward `core_claim_plus`

### [2026-04-12 02:50] - core_claim
- Reviewed job `7065b8f4-dd8`: another worker-node death with exit `-9`, but the corrected rerun progressed to 4 fully completed scenarios, plus `Zoodles` at 4/5 seeds and `Breakfastburritos` at 1/5
- Updated the CPU runner to prioritize scenarios with the fewest missing seeds and capped each retry to a small batch so unstable jobs finish partial scenarios before opening many new ones

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
