# Research Direction

## Research Problem
The target problem is recovering prerequisite structure between procedural steps from repeated demonstrations of the same task. Future work should still operate in this problem family: infer graphs from procedure executions and improve graph quality on CaptainCook4D.

## Approach Scope
Scientists may explore alternative graph-learning methods, training strategies, calibration schemes, or graph post-processing procedures, as long as the output remains a comparable predicted prerequisite graph for each procedure.

## Out of Bounds
- Using additional datasets, extra annotations, or external knowledge sources that make the CaptainCook4D comparison no longer like-for-like.
- Changing the scored task from graph prediction to some other objective.
- Reporting results outside the provided evaluation contract instead of `scores.json`.
