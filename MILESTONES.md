# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10] - method_runs
- Paper: Task Graph Maximum Likelihood (TGML) for learning task graphs from procedural activity videos
- Code implementation complete: TGML loss, DO model, Count-Based baseline
- Container switch: scratch bootstrap failed due to fakeroot shared library issue; switching to Docker bootstrap (python:3.12-bookworm) for full OS support
- Local validation: Count-Based baseline F1=61.4 (paper: 60.2) ✓
- Local validation: DO training works (Coffee scenario, 50 epochs, seq_acc 0.405→0.994) ✓
- DO evaluation works: Coffee scenario F1=88.9 after 50 epochs ✓
- Submitting GPU job: 24 scenarios × 5 seeds × 1000 epochs on CaptainCook4D

## Stop Justification

