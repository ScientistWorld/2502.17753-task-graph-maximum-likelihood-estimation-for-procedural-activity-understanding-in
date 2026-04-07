# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results.

## Metrics

Task graph generation is evaluated by comparing the predicted binary, unweighted graph against the ground truth graph.

Given the predicted adjacency set $\hat{\mathcal{A}}$ and ground truth adjacency set $\mathcal{A}$:
- **Precision** = $|\hat{\mathcal{A}} \cap \mathcal{A}| / |\hat{\mathcal{A}}|$ — fraction of predicted edges that are correct
- **Recall** = $|\hat{\mathcal{A}} \cap \mathcal{A}| / |\mathcal{A}|$ — fraction of ground truth edges that are recovered
- **F1** = $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$ — harmonic mean of precision and recall

All metrics are reported as percentages (0-100). Higher is better for all three.

## Baselines and Targets

The paper compares against multiple baselines:

**Task Graph Generation (F1 score):**
- MSGI: 12.2% (CaptainCook4D), 50.6% (EgoPER), 22.9% (EgoProceL)
- Count-Based: 60.2% (CaptainCook4D), 80.8% (EgoPER), 49.5% (EgoProceL)
- MSG2: 73.3% (CaptainCook4D), 67.5% (EgoPER), 55.4% (EgoProceL)

The paper's proposed methods achieve:
- DO: ~87.8% F1 (CaptainCook4D), ~91.0% F1 (EgoPER), ~72.3% F1 (EgoProceL)
- TGT-text: ~80.8% F1 (CaptainCook4D), ~85.0% F1 (EgoPER), ~66.9% F1 (EgoProceL)

For mistake detection: F1 score on detecting correct vs. mistake actions (average F1 across classes).

## Evaluation Protocol

1. **Dataset**: The paper evaluates task graph generation on CaptainCook4D (24 cooking procedures), EgoPER (5 procedures), and EgoProceL (13 procedures). Each procedure has annotated action sequences and a ground-truth task graph used only for evaluation.

2. **Per-procedure learning**: Models are learned separately for each procedure/task graph from that procedure's action sequences. The final reported score is the mean across procedures in the dataset.

3. **Weak supervision**: During training, models observe only action sequences. Ground-truth graphs are not used for optimization.

4. **Evaluation**: After learning, the predicted adjacency matrix is thresholded at `1/n`, post-processed into a valid DAG, and then compared against the ground-truth graph with precision, recall, and F1 over directed edges.

5. **Reporting**: Results are averaged across all procedures in the dataset. The paper reports mean and standard deviation over 5 random seeds for their method, and single numbers for baselines (deterministic methods).

6. **Core reproduction target**: CaptainCook4D is the primary benchmark. The paper's main claim is that the proposed DO method significantly outperforms prior work on this dataset. Reproducing the DO results on CaptainCook4D is the core claim to validate.
