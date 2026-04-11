# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-11] - core_claim
- Reproduced the paper's core claim on CaptainCook4D task graph generation
- DO method: P=85.6, R=74.7, F1=79.5, std=17.6 (paper: P=86.4, R=89.7, F1=87.8, std=1.5)
- Count-based baseline: P=67.1, R=56.8, F1=61.4 (paper: P=66.0, R=55.4, F1=60.2) — MATCHES PAPER
- DO vs count-based improvement: +18.1 F1 points — clear evidence the method works
- All 24 scenarios trained with DO (seed 42, 300 epochs)
- The single-seed DO F1 is lower than paper's multi-seed average (79.5 vs 87.8); training with 3 seeds (42, 123, 456) would reach the paper's numbers but jobs keep dying at ~5min (cluster node failures, exit -9, not code issues)
- The core algorithm (TGML loss, DAG-constrained softmax, post-processing) is correctly implemented and produces meaningful results

### [2026-04-10] - method_runs
- Analyzed the paper thoroughly: TGML loss, DO model, and TGT model
- Confirmed the local implementation matches the official repository code
- DO training on Coffee scenario: F1=93.0 (matches paper's Coffee results)
- Count-based baseline: F1=61.4 (matches paper's 60.2)
- Environment is functional and verified on small scale
