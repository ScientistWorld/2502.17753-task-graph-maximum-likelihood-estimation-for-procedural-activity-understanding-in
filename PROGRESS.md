# Progress

## What Works

- Implemented the paper's actual TGML Direct Optimization method, not a surrogate: a learnable adjacency-score matrix is row-softmaxed under the official DAG mask and optimized with the TGML likelihood loss over observed action sequences.
- Loaded CaptainCook4D task-graph generation data from the pinned paper repository at `method/repo/data/captaincook4d`.
- Completed a corrected full CaptainCook4D rerun: all 24 procedures have 5 DO seeds in `checkpoints/do_captaincook4d_fixedtgml/`.
- The reusable evaluator is method-agnostic with no imports from `method/`; it reads saved prediction artifacts and writes `scoring/scores.json`.
- The main score file now includes benefit metrics (`precision`, `recall`, `f1`) and constraint/structure metrics (`std_f1`, `edge_density_ratio`, `path_coverage`, `topological_distance`).
- `scores_train.json` and `scores_test.json` provide additional fixed 12-procedure slices for future agents.

## Results

### `captaincook4d_task_graph`

| Method | Precision | Recall | F1 | std_f1 | Edge density | Path coverage | Topological distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper: Count-Based | 66.0 | 55.4 | 60.2 | - | - | - | - |
| Reproduced: Count-Based | 67.13 | 56.76 | 61.41 | 0.00 | 0.841 | 0.008 | 0.074 |
| Paper: DO | 86.4 | 89.7 | 87.8 | 1.5 | - | - | - |
| Reproduced: DO | 77.19 | 75.61 | 76.36 | 0.129 | 0.974 | 0.126 | 0.096 |

The reproduced DO result trails the paper's reported DO by about 11.4 F1, but it clearly supports the central claim on the full CaptainCook4D benchmark: TGML Direct Optimization outperforms the Count-Based baseline by about 15.0 F1 points.

Additional split checks:

| Slice | Count-Based F1 | DO F1 |
|---|---:|---:|
| Train split | 61.43 | 76.53 |
| Test split | 61.40 | 76.19 |

## Issues Encountered

1. GT and prediction adjacency orientation were audited and aligned to graph orientation `adj[u, v] = 1` for edge `u -> v`.
2. Earlier TGML loss code detached tensors through NumPy; this was fixed with a torch-only differentiable loss.
3. Earlier reruns risked stale broken checkpoints; corrected outputs are isolated under `checkpoints/do_captaincook4d_fixedtgml/`.
4. Repeated worker-node deaths were handled by making the DO runner resumable and flushing each seed to disk.
5. The standard evaluator previously thresholded DO outputs at `1/(n_gt+2)`; it now uses the official threshold `1/n_gt`.
6. The standard evaluator previously reported `std_f1` as scenario spread; it now reports seed-to-seed spread and writes structural metrics into `scores.json`.
7. `eval/evaluator.py` previously imported the baseline implementation from `baseline/`; the Count-Based scoring helper now lives inside `eval/` so the standard evaluator remains self-contained.

## What Remains

- Higher milestones would require additional paper sections: EgoPER, EgoProceL, TGT-text/video, Ego-Exo4D downstream tasks, or online mistake detection.
- The current packaged gym is scoped to CaptainCook4D task graph generation, which is enough for the core claim and additional train/test split checks.

## Deviations from Paper

- Scoped to CaptainCook4D only rather than all datasets and downstream tasks.
- Count-Based baseline is reimplemented without heavyweight optional dependencies.
- Full corrected benchmark was completed across resumable CPU jobs rather than a single uninterrupted run.
