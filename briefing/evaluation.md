# Evaluation

## What is scored

Two experiments, each a set of procedures with one ground-truth prerequisite graph per procedure:

| Experiment | Public procedures (dev) | Held-out procedures (hidden test) | Weight |
|---|---:|---:|---:|
| `captaincook4d_task_graph` | 12 | 12 | 0.7 |
| `noisy_demonstrations_task_graph` | 100 | 100 | 0.3 |

Demonstrations of every procedure, including the held-out ones, are available to the learner. Ground-truth graphs are available only for the public procedures; held-out graphs never leave the hidden evaluator.

- CaptainCook4D data: `data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d/` (fetched by `scripts/download.sh`). For held-out procedures the graph files contain only the node list (`steps`); `edges` are withheld.
- Noisy-demonstration data: `data/noisy_demos/train.json` (public procedures, with `edges`) and `data/noisy_demos/test_inputs.json` (held-out procedures, demonstrations only).
- `data/loader.py` loads both (`load_data`, `load_noisy_demos`).

Node ids: 0 = START, n-1 = END, key-steps 1..n-2. A JSON edge `[a, b]` means "a is a precondition of b".

## Prediction contract

A method is a directory `checkpoints/<method_name>/`. For every procedure it predicts, it writes `<procedure>_predictions.npz`. CaptainCook4D procedure names are lowercase, e.g. `coffee`; noisy procedures are `noisytrain_00`... and `noisytest_00`.... The file contains

- `pred_cores`: float array of shape **(5, n, n)**. These are five independent trials (e.g. five seeds); a deterministic method repeats its matrix five times. Entry `[i, j]` scores "node j is a precondition of node i" and must be **finite and in [0, 1]**.

Every directory under `checkpoints/` that contains at least one `*_predictions.npz` is scored as a method, keyed by its directory name. For each experiment, a method must cover **every** procedure of that experiment. A missing file, wrong shape, non-finite or out-of-range value marks the method as **failed** on that experiment, and a failed method receives the floor of every metric. Seed ids, score files, registries or any other numbers a method writes are never read.

## Post-processing (fixed by the evaluator)

Each trial is converted into a binary DAG exactly as in the reference protocol:
1. binarise at the fixed threshold `1/n`;
2. connect END to every node without dependents and every node without preconditions to START;
3. transitive reduction;
4. while a cycle remains, delete its lowest-scored edge.

## Metrics

All metrics are computed per trial, averaged over the five trials, then averaged over the procedures of an experiment.

| Metric | Direction | Coefficient | Meaning |
|---|---|---:|---|
| `f1` | higher | 1.0 | F1 (%) of predicted directed edges vs ground-truth edges (primary) |
| `precision` | higher | 0.25 | edge precision (%) |
| `recall` | higher | 0.25 | edge recall (%) |
| `worst_quartile_f1` | higher | 0.35 | mean F1 over the weakest quarter of procedures |
| `edge_density_ratio` | higher | 0.15 | min(#pred, #gt edges) / max(#pred, #gt edges) |
| `path_coverage` | higher | 0.12 | fraction of ground-truth START-END paths whose every edge is predicted |
| `topological_distance` | lower | 0.18 | fraction of ordered node pairs whose precedence (reachability) differs |

The evaluator also scores its own Count-Based baseline (key `count_based`) from the same demonstrations.

**Ranking.** For each experiment and metric, the improvement over the reference is taken in the metric's better direction. It is multiplied by the coefficient's magnitude and divided by the metric's spread across ranked attempts. These terms are summed over metrics and weighted by the experiment weight.

## Scripts

- `bash scripts/evaluate.sh`: public procedures, writes `scoring/scores.json`.
- `bash scripts/evaluate_train.sh`: public procedures, writes `scoring/scores_train.json` (same numbers).
- `bash scripts/evaluate_test.sh`: hidden test on held-out procedures, writes `scoring/scores_test.json`.

None of them needs arguments, and none retrains anything.

## Reference scale

On CaptainCook4D the strongest prior method reported for this benchmark reaches about 73 F1 and the best reported result about 88 F1 (all 24 procedures). The Count-Based baseline is around 61 F1. Near-perfect scores would be suspicious on this benchmark.
