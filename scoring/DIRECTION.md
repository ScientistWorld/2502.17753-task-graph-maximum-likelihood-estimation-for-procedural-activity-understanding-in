# Research Direction

## Research problem
Recover prerequisite structure between procedural steps from a few demonstrations of the same task, including when demonstrations are noisy. Methods are ranked on held-out procedures of both experiments: CaptainCook4D (weight 0.7) and noisy demonstrations (weight 0.3).

## Open directions
- Separating real prerequisites from orders that are fixed only by coincidence in few demonstrations (precision on CaptainCook4D).
- Robustness to inserted, deleted and mislabelled steps without losing accuracy on clean data. A method tuned only for clean data and one tuned only for noisy data each give up one experiment.
- Calibration of the weighted adjacency against the fixed 1/n threshold, and principled model selection on the public procedures.
- Stability across trials and across procedures (`worst_quartile_f1`).

## Out of bounds
- Additional datasets, annotations, language or step-name knowledge, or any source of held-out graphs.
- Changing the evaluator, its data, threshold or post-processing.
- Changing the task from graph prediction to another objective, or reporting results outside the provided evaluators.
