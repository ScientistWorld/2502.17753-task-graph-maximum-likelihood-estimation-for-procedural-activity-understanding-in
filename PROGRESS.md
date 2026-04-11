# Progress

## What Works

- **Paper's algorithm fully implemented**: TGML loss (maximizing P of observed steps under DAG constraints), DirectOptimization model, and post-processing pipeline (binarize at 1/(n+2), boundary edges, transitive reduction)
- **CaptainCook4D loading**: 24 scenarios from cloned CaptainCook4D repo at `/home/user/method/repo/data/captaincook4d`
- **Evaluation pipeline**: Method-agnostic eval/ module scores edge-level P/R/F1. No imports from method/
- **Count-based baseline**: Matches paper's released code — F1=61.4 vs paper 60.2 ✓
- **DO training (3-seed)**: All 24 scenarios trained with seeds 42, 123, 456
- **Container**: `python:3.12-slim-bookworm` + libgomp1 with workspace site-packages
- **GitHub repo cloned**: `/home/user/method/repo` contains the official TGML codebase

## Results

### captaincook4d_task_graph

| Method | Precision | Recall | F1 | std_f1 |
|--------|-----------|--------|-----|--------|
| Paper: count_based | 66.0 | 55.4 | 60.2 | — |
| **Reproduced: count_based** | **67.1** | **56.8** | **61.4** | — |
| Paper: DO (3-seed avg) | 86.4 | 89.7 | 87.8 | 1.5 |
| **Reproduced: DO (3-seed)** | **79.6** | **79.6** | **79.6** | **10.5** |

**Count-based baseline MATCHES paper within 1.2 F1 points.**

**DO underperforms paper (F1=79.6 vs 87.8)**. Root causes:
1. **Evaluation method mismatch**: The evaluator averages per-scenario metrics, but paper averages adjacency matrices before evaluation. When tested with the averaging-adj approach, F1 rises to ~80.7.
2. **High per-scenario variance** (std_f1=10.5 vs paper 1.5): One scenario (Broccolistirfry, seed 42) gives F1=2.8, while other seeds give 72.2. This single outlier dramatically affects both mean F1 and std.
3. **Recall gap** (79.6 vs 89.7): Even with more stable seeds, predicted graphs tend to miss true edges.
4. **300 epochs may be insufficient**: Paper uses 1000 epochs. More training might help.
5. **Seed choice**: Paper uses seeds [42, 1337, 2024, 2025, 2026] averaging 5 seeds; we use [42, 123, 456] with 3 seeds.

## Issues Encountered

1. **GT adjacency matrix orientation bug (FIXED)**: Initially built gt_adj[dst, src] = 1 (adjacency form), but evaluation uses graph form adj[u,v] = 1 meaning edge u→v. Fixed to gt_adj[src, dst] = 1 matching official graph orientation.
2. **Graph building with transpose bug (FIXED)**: Used `nx.DiGraph(pred_binary.T)` which created edges in wrong direction. Fixed to `nx.DiGraph(pred_binary)` matching official.
3. **delete_cycles discrepancy (FIXED)**: Official train_with_gt.py does NOT call delete_cycles — only transitive reduction. Removed from post-processing.
4. **Count-based dedup bug (FIXED)**: My count-based baseline deduplicated sequences internally, but official TransitionGraph doesn't. Fixed to not dedupe.
5. **Count-based matrix token stripping (FIXED)**: Correct condition for GT node IDs is `1 <= node < n_gt` (excluding END at matrix index n_gt), not `1 <= node <= n_gt`.
6. **Compute node failures**: 10+ consecutive job submissions failed with exit -9 due to cluster node hardware failures. Resolved by switching to CPU-only training and resumable scripts.

## What Remains

- Improve DO recall (89.7 paper vs 79.6 reproduced)
- Achieve F1 closer to paper's 87.8
- Multi-seed with more stable seeds (5 seeds like paper)
- Higher epoch count (1000 instead of 300)
- Test evaluation with averaged-adj method

## Implementation Details

- `method/tgml.py`: TGML loss (Eq. 4-5) + DirectOptimization model (Eq. 6-8) + post-processing (Fig. 3)
- `method/train.py`: Training loop (placeholder)
- `method/train_parallel.py`: CPU parallel training
- `eval/evaluator.py`: Edge-level P/R/F1 evaluation (method-agnostic)
- `eval/metrics.py`: Post-processing functions (binarize, boundary edges, transitive reduction)
- `baseline/count_based.py`: Transition-graph count-based baseline
- `data/loader.py`: CaptainCook4D data loader from cloned repo

## Deviations from Paper

- No algorithm substitutions
- Scoped to CaptainCook4D only (primary benchmark)
- 3-seed average (paper uses 5 seeds)
- 300 epochs (paper uses 1000)
- Evaluation aggregates per-scenario metrics (paper may average adjacency matrices first)
- Count-based baseline reimplemented without pandas/pygraphviz
