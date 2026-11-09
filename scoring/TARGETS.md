# Evaluation Targets

Two scored experiments, both reported in `scores_train.json` / `scores.json` (public procedures) and `scores_test.json` (hidden test, held-out procedures). Every target below has an entry for each experiment in every score file.

## Experiment `captaincook4d_task_graph` (weight 0.7)
CaptainCook4D task-graph generation: 12 public and 12 held-out procedures, five trials per procedure.

## Experiment `noisy_demonstrations_task_graph` (weight 0.3)
Task graphs from demonstrations perturbed by insert/delete/replace noise (p = 0.05): 100 public and 100 held-out procedures, five trials per procedure.

## Primary target (benefit)
- **`f1`**: edge F1 against the ground-truth graph. Improve it on both experiments.

## Secondary quality signals (benefit)
- **`precision`**: do not raise F1 by adding spurious prerequisite edges.
- **`recall`**: do not raise F1 by returning an incomplete graph.

## Constraint targets
- **`worst_quartile_f1`**: mean F1 of the weakest quarter of procedures. Do not trade hard procedures for easy ones.
- **`edge_density_ratio`**: predicted and ground-truth edge counts should match. This blocks over- and under-connected graphs.
- **`path_coverage`**: complete ground-truth START-END chains should be recovered.
- **`topological_distance`** (lower is better): the predicted partial order (reachability) should agree with the ground truth.
- **Validity (hard constraint)**: five finite trials in [0, 1] for every procedure of an experiment. Otherwise the run is scored as failed: floor value on every metric.

Seed-to-seed spread is reported only as an unscored diagnostic (`diagnostics.<exp>.<method>.trial_std_f1`), because the evaluator cannot verify that trials come from different seeds.

## Reference values (hidden test, `scores_test.json`)

| Experiment | Method | Precision | Recall | F1 | Worst-quartile F1 | Edge density | Path coverage | Topo. distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| captaincook4d | reference (`do`) | 87.26 | 89.85 | 88.35 | 76.46 | 0.953 | 0.582 | 0.016 |
| captaincook4d | `count_based` | 67.33 | 56.63 | 61.40 | 48.26 | 0.838 | 0.017 | 0.073 |
| noisy demos | reference (`do`) | 64.56 | 72.23 | 68.05 | 50.91 | 0.885 | 0.147 | 0.112 |
| noisy demos | `count_based` | 68.13 | 59.34 | 63.36 | 46.40 | 0.868 | 0.033 | 0.135 |

On the public procedures (`scores_train.json`) the reference scores 88.55 F1 on CaptainCook4D and 66.98 on noisy demonstrations; Count-Based scores 61.43 and 62.65. The paper reports DO precision/recall/F1 of 86.4 / 89.7 / 87.8 and Count-Based 60.2 F1 on all 24 CaptainCook4D procedures.

## Required comparison (the paper's claim)
The reference must beat the Count-Based baseline on F1 by a clear margin on CaptainCook4D. It does: +26.95 F1 on the held-out procedures, and better on all 12. It also wins on noisy demonstrations: +4.69 F1, paired 95% CI [+0.93, +8.41].

## Headroom
No scored metric of the reference is at its bound. Honest alternatives already score higher:
- a demonstration-consistency heuristic: +1.22 F1 on CaptainCook4D, +9.43 on noisy demonstrations;
- choosing the paper's β on the public procedures: +13.11 F1 on noisy demonstrations.

`HARDENING.md` gives the full audit.

## How to run
- `bash scripts/method.sh`: train the reference into `checkpoints/do/` (CPU, about 15 minutes).
- `bash scripts/evaluate.sh` / `bash scripts/evaluate_train.sh`: score every `checkpoints/*/` method on public procedures.
- `bash scripts/evaluate_test.sh`: hidden test, no arguments.
- `python validate.py`.
