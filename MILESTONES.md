# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-07 09:31] - method_runs
- Fixed a baseline-evaluation bug where `eval/evaluator.py` was post-processing the Count-Based adjacency a second time, collapsing the orientation and producing 0.0 F1
- Re-verified locally that the corrected Count-Based path scores about 67.1 precision, 56.8 recall, and 61.4 F1 on CaptainCook4D
- Tightened `TARGETS.md`, `CHECKLIST.md`, and briefing docs so they reflect the actual scoped CaptainCook4D experiment that the next submission will score
- Kept the milestone at `method_runs` because fresh DO results from the corrected pipeline are still pending the submitted job

### [2026-04-07 09:36] - method_runs
- The first submit attempt failed during container build because `environment/container.def` used `Bootstrap: library`
- Switched the container base to `Bootstrap: docker` with the same `ubuntu:22.04` image so Apptainer does not depend on an unavailable library remote
- Kept the milestone at `method_runs`; no experiment results changed, this is only an environment fix before resubmission

### [2026-04-07 00:00] - method_runs
- Replaced the simplified local count-based baseline with a transition-graph implementation that matches the paper's released baseline logic closely
- Verified on CaptainCook4D annotations that the corrected baseline reaches about 61.4 mean F1 locally, which is on the same scale as the paper's reported 60.2 baseline
- Kept the milestone at `method_runs` because the proposed DO result still needs a fresh run after the baseline-fidelity correction

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
