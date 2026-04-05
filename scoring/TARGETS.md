# Evaluation Targets

## Primary: task_graph_edge_prediction_f1
Task graph edge prediction F1 score on CaptainCook4D (24 cooking procedures).
The model predicts directed edges between task steps from action sequences, and F1 measures
how well the predicted prerequisite graph matches the ground truth.
- **Metric**: f1 (higher is better)
- **How to run**: `python method/train.py --dataset captaincook4d --repo_data_path method/repo/data/captaincook4d --output checkpoints/`

## Constraint: recall_above_baseline
Recall must remain above the count-based baseline (~54%), ensuring the method
discovers genuine edges rather than overfitting to precision by predicting too few edges.
- **Metric**: recall (higher is better, must exceed count-based baseline)

## Ablation: vs_count_baseline
F1 improvement over the count-based co-occurrence baseline demonstrates that the
TGML loss captures graph structure beyond simple statistical patterns.
- **Metric**: f1_gap (DO F1 minus count-based F1, higher is better)
