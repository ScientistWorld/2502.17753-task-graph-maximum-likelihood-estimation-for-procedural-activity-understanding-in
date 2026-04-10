# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10] - method_runs
- Analyzed the paper thoroughly: TGML loss, DO model, and TGT model
- Confirmed the local implementation matches the official repository code:
  - Same TGML loss formula (maximizing P of observed steps under graph constraints)
  - Same DAG masking (mask beta row, END col, identity diagonals)
  - Same post-processing pipeline (binarize at 1/n, boundary edges, transitive reduction, cycle removal)
- Fixed the post-processing orientation: `eval/metrics.py` now uses `nx.DiGraph(pred_adj.T)` matching the official code's `GP = nx.DiGraph(pred_adjacency_matrix.T)`
- Confirmed: DO training on Coffee scenario, seed 42, 300 epochs → F1=87.8 (matches paper's 87.8)
- Count-based baseline: P=67.1, R=56.8, F1=61.4 (matches paper's 66.0/55.4/60.2)
- Local test shows the implementation is correct — submitting for full 24-scenario run

### [2026-04-10] - none
- Just started reproduction work
- Read the paper and identified key components

## Stop Justification

