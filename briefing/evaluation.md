# Evaluation

## Metrics

Task graph generation is evaluated by comparing a predicted binary prerequisite graph against a ground-truth prerequisite graph for the same procedure.

Given the predicted adjacency set $\hat{\mathcal{A}}$ and ground truth adjacency set $\mathcal{A}$:
- **Precision** = $|\hat{\mathcal{A}} \cap \mathcal{A}| / |\hat{\mathcal{A}}|$ — fraction of predicted edges that are correct
- **Recall** = $|\hat{\mathcal{A}} \cap \mathcal{A}| / |\mathcal{A}|$ — fraction of ground truth edges that are recovered
- **F1** = $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$ — harmonic mean of precision and recall

All metrics are reported on directed edges. Higher is better for precision, recall, and F1.

## Benchmark Scope

The paper reports graph-generation results on three datasets, with CaptainCook4D as the main benchmark. This workspace is currently packaged around the CaptainCook4D experiment because it is the paper's clearest central-claim setting and is the one reproduced here.

## Evaluation Protocol

1. **Dataset**: CaptainCook4D contains 24 cooking procedures. Each procedure has multiple annotated action sequences and one ground-truth task graph.
2. **Per-procedure prediction**: Produce one predicted graph per procedure from that procedure's action sequences.
3. **Weak supervision**: Graph annotations are not part of the learning signal for the comparable setting; they are used only for evaluation.
4. **Graph cleanup before scoring**: Convert the raw graph output into a binary DAG-compatible prediction, then compare its edges against the ground-truth graph.
5. **Dataset score**: Compute precision, recall, and F1 for each procedure, then report the mean across procedures.

## Reference Scale

On CaptainCook4D, the strongest prior baseline in the paper reaches about 73 F1, while the paper's best reported result reaches about 88 F1. Reproduced scores should stay on that difficulty scale; near-perfect performance would be suspicious for this benchmark.
