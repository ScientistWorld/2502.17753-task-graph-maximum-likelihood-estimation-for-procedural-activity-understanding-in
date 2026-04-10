# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10] - method_runs
- Paper: Task Graph Maximum Likelihood (TGML) for learning task graphs from procedural activity videos
- Code implementation complete: TGML loss, DO model, Count-Based baseline
- Previous GPU jobs failed (Docker bootstrap OOM, scratch bootstrap builder failures)
- Local validation with vendored Python (torch 2.11 + numpy + networkx): Count-Based F1=61.4 ✓, DO training works ✓
- Environment rebuilt with full scratch runtime: python3 + torch + numpy + networkx + uv + uvx all vendored
- All imports verified: data loader, evaluator, metrics, DO trainer, count-based baseline
- Submitting GPU job: 24 scenarios × 5 seeds × 1000 epochs on CaptainCook4D

## Stop Justification

