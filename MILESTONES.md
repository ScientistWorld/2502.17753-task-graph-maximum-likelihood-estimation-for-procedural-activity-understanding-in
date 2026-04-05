# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-05 03:55] - method_runs
- Audited the local DO/TGML pipeline against the released repository bundled in `method/repo`
- Confirmed the implementation is the paper's algorithm, but found fidelity bugs that block a justified `core_claim`
- Fixed CaptainCook4D preprocessing to prepend `START`, aligning training sequences with the official DO code
- Fixed the DO adjacency mask to match the released implementation and aligned boundary-edge post-processing with the paper
- Removed the invalid evaluation fallback that silently scored the count-based baseline as `do`
- Recompiled the patched Python modules and verified the loader now emits `[beta, START, ... , END, gamma]` sequences on CaptainCook4D

## Stop Justification
