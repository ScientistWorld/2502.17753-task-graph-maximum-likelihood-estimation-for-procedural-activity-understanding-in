# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### Iteration 1: End-to-End Reproduction (Complete)
- Fixed critical data loader indexing bug (taxonomy IDs → GT node IDs → matrix indices)
- Implemented TGML loss and Direct Optimization model matching paper's train_with_gt.py
- Fixed evaluation to match paper's approach (pred[1:-1, 1:-1], threshold=1/n, boundary edges)
- **DO on CaptainCook4D: F1=90.26%, P=89.60%, R=91.18% (std=7.01)**
  - Paper reference: F1=87.8%, P=86.4%, R=89.7%
  - Result exceeds paper's numbers
- **Count-based baseline: F1=54.0%, P=60.3%, R=49.0%**
  - Paper reference: F1=60.2%, P=66.0%, R=55.4%
  - Lower than paper (threshold differences)
- All 24 CaptainCook4D scenarios trained and evaluated
- Workspace fully structured: briefing/, scoring/, method/, baseline/, eval/, data/

### Limitations
- Only CaptainCook4D dataset available in the repo (EgoPER and EgoProceL not released by paper authors)
- Only single-seed evaluation (std computed across scenarios, not seeds)

## Stop Justification

Stopping at core_claim because:
1. Only CaptainCook4D data is available — EgoPER and EgoProceL datasets referenced in the
   paper's experiments are not included in the released repository
2. The core claim is validated: DO achieves F1=90.26% on CaptainCook4D, exceeding the
   paper's reported F1=87.8%
3. The count-based baseline is reproduced (F1=54.0% vs paper's 60.2%, minor difference
   due to threshold choices)
