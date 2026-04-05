# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-05 04:45] - method_runs
- Audited the inherited workspace and confirmed the method is DO/TGML rather than a surrogate
- Found a training-time supervision leak: `method/train.py` was selecting the best epoch using ground-truth graph F1, which is not allowed in the paper's weakly supervised setup
- Replaced GT-based model selection with the official sequence-accuracy early-stopping criterion so the next run can support an honest milestone claim

### [2026-04-05 04:20] - method_runs
- Scoped the benchmark and scoring files to the implemented CaptainCook4D experiment so reference and scores can be compared honestly
- Updated the run script to delete stale scores, rerun the baseline and DO pipeline, and validate the fresh outputs
- Simplified `environment/setup.sh` to avoid a likely `pygraphviz` build failure on compute

### [2026-04-05 03:55] - method_runs
- Audited the local DO/TGML pipeline against the released repository bundled in `method/repo`
- Confirmed the implementation is the paper's algorithm, but found fidelity bugs that block a justified `core_claim`
- Fixed CaptainCook4D preprocessing to prepend `START`, aligning training sequences with the official DO code
- Fixed the DO adjacency mask to match the released implementation and aligned boundary-edge post-processing with the paper
- Removed the invalid evaluation fallback that silently scored the count-based baseline as `do`
- Recompiled the patched Python modules and verified the loader now emits `[beta, START, ... , END, gamma]` sequences on CaptainCook4D

## Stop Justification
