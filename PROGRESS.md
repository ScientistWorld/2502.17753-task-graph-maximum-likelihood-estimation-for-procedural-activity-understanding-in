# Progress

## What Works

- **Paper's algorithm fully implemented**: TGML loss (maximizing P of observed steps under DAG constraints), DirectOptimization model, and post-processing pipeline (binarize at 1/(n+2), boundary edges, transitive reduction)
- **CaptainCook4D loading**: 24 scenarios from cloned CaptainCook4D repo at `/home/user/method/repo/data/captaincook4d`
- **Evaluation pipeline**: Method-agnostic eval/ module scores edge-level P/R/F1. No imports from method/
- **Count-based baseline**: Matches paper's released code — F1=61.4 vs paper 60.2 ✓
- **DO training path repaired**: TGML loss now preserves autograd through the adjacency matrix, and a minimal smoke test confirms nonzero gradients and parameter updates
- **Container**: `python:3.12-slim-bookworm` + libgomp1 with workspace site-packages
- **GitHub repo cloned**: `/home/user/method/repo` contains the official TGML codebase

## Results

### captaincook4d_task_graph

| Method | Precision | Recall | F1 | std_f1 |
|--------|-----------|--------|-----|--------|
| Paper: count_based | 66.0 | 55.4 | 60.2 | — |
| **Reproduced: count_based** | **67.1** | **56.8** | **61.4** | — |
| Paper: DO (5-seed avg) | 86.4 | 89.7 | 87.8 | 1.5 |

**Count-based baseline MATCHES paper within 1.2 F1 points.**

**The previously written DO scores are not treated as valid evidence now.**
This turn uncovered that the local TGML loss had been detaching the adjacency matrix to NumPy, which breaks gradient flow. The stored DO numbers in `scoring/scores.json` were produced before fixing that bug, so they are retained only as stale artifacts and will be overwritten by the next submission.

## Issues Encountered

1. **GT adjacency matrix orientation bug (FIXED)**: Initially built gt_adj[dst, src] = 1 (adjacency form), but evaluation uses graph form adj[u,v] = 1 meaning edge u→v. Fixed to gt_adj[src, dst] = 1 matching official graph orientation.
2. **Graph building with transpose bug (FIXED)**: Used `nx.DiGraph(pred_binary.T)` which created edges in wrong direction. Fixed to `nx.DiGraph(pred_binary)` matching official.
3. **delete_cycles discrepancy (FIXED)**: Official train_with_gt.py does NOT call delete_cycles — only transitive reduction. Removed from post-processing.
4. **Count-based dedup bug (FIXED)**: My count-based baseline deduplicated sequences internally, but official TransitionGraph doesn't. Fixed to not dedupe.
5. **Count-based matrix token stripping (FIXED)**: Correct condition for GT node IDs is `1 <= node < n_gt` (excluding END at matrix index n_gt), not `1 <= node <= n_gt`.
6. **Compute node failures**: 10+ consecutive job submissions failed with exit -9 due to cluster node hardware failures. Resolved by switching to CPU-only training and resumable scripts.
7. **Training selection leakage risk (FIXED IN SCRIPTS)**: `scripts/method.sh` previously selected the best epoch using GT F1 during training. Updated it to save only the final fixed-epoch model, keeping training weakly supervised.
8. **Autograd break in TGML loss (FIXED)**: `method/tgml.py` detached the adjacency matrix to NumPy inside the loss, so `loss.backward()` could not update the graph parameters. Replaced this with a torch-only implementation and trimmed padded sequences at the first `gamma` token.

## What Remains

- Rerun CaptainCook4D DO from scratch with the corrected differentiable loss
- Produce fresh `scores.json` before claiming `core_claim`
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
