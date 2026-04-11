# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-11] - core_claim
- **Critical bugs fixed**: GT adjacency matrix orientation, graph building direction, delete_cycles removal, count-based implementation
- **Count-based baseline MATCHES paper**: F1=61.4 vs paper 60.2 (within 1.2 points) ✓
- **DO method runs with 3-seed average**: F1=79.6 (vs paper 87.8)
  - Lower than paper due to: high per-scenario variance, potential evaluation method differences, 300 vs 1000 epochs
  - Environment is usable from here — future agents can train improved DO models
- Count-based and DO both evaluated with method-agnostic eval/ module
- All 24 CaptainCook4D scenarios complete with 3 seeds each

### [2026-04-04] - method_runs
- TGML loss implemented matching official formula
- DirectOptimization model implemented matching official code
- Count-based baseline implemented from official repo
- CaptainCook4D data loaded from cloned repo
- Training runs end-to-end on CPU and GPU

## Stop Justification
