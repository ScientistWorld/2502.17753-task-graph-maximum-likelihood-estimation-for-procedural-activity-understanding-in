# Progress

## What Works

- The paper's TGML Direct Optimization is implemented, not a surrogate. `method/tgml.py` holds the masked row-softmax adjacency and the TGML loss (Eq. 11), plus an exact batched form. `method/train.py` holds the paper's training protocol: Adam with lr 0.1, beta 0.005 (0.5 for a procedure without any complete demonstration), at most 1000 epochs, and Sequence-Accuracy early stopping with patience 25. The state at the halting epoch is the output, as in the released `train_with_gt.py`.
- `bash scripts/method.sh` trains five seeds on every procedure of both scored experiments into `checkpoints/do/` (CPU, about 15 minutes). The result is bit-for-bit deterministic.
- `eval/train/` (public procedures) and `eval/test/` (hidden test, held-out procedures) are self-contained and method-agnostic. They score every `checkpoints/*/` directory plus their own Count-Based baseline, and reject invalid artifacts.
- `scripts/download.sh` fetches the pinned CaptainCook4D data as a tarball and withholds the held-out graphs.

## Results (hidden test, held-out procedures)

| Experiment | Method | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| CaptainCook4D (paper, 24 procedures) | Count-Based | 66.0 | 55.4 | 60.2 |
| CaptainCook4D (paper, 24 procedures) | DO | 86.4 | 89.7 | 87.8 |
| CaptainCook4D (12 held-out) | Count-Based | 67.33 | 56.63 | 61.40 |
| CaptainCook4D (12 held-out) | DO (reference) | 87.26 | 89.85 | 88.35 |
| Noisy demonstrations (100 held-out) | Count-Based | 68.13 | 59.34 | 63.36 |
| Noisy demonstrations (100 held-out) | DO (reference) | 64.56 | 72.23 | 68.05 |

Public procedures: DO scores 88.55 F1 on CaptainCook4D and 66.98 on noisy demonstrations; Count-Based scores 61.43 and 62.65.

## History

- Earlier iterations omitted the paper's early stopping and reported 76.36 F1 (24 procedures). The final-iteration hardening restored it; see `HARDENING.md` for the audit of variants and shortcuts.

## Deviations from Paper

- Scoped to CaptainCook4D task-graph generation. EgoPER, EgoProceL, TGT and the downstream tasks are not reproduced.
- Results are reported on two fixed 12-procedure halves rather than on all 24 procedures.
- The noisy-demonstration experiment is an added, semi-synthetic benchmark for headroom; the paper has no numbers for it.
- The Count-Based baseline is a reimplementation built into the evaluators.
