# Verification

## Current Verified State

- `python3 validate.py --compare` passes.
- `scoring/scores.json` reflects the full 24-procedure CaptainCook4D benchmark.
- `checkpoints/do_captaincook4d_fixedtgml/` contains 24 prediction files, each with 5 DO seeds.
- `scoring/scores_train.json` and `scoring/scores_test.json` provide additional 12-procedure split checks.

## Key Numbers

- Count-Based: precision 67.13, recall 56.76, F1 61.41.
- TGML DO: precision 77.19, recall 75.61, F1 76.36.
- TGML DO seed-to-seed `std_f1`: 0.129.

## Commands Run

```bash
bash scripts/evaluate.sh captaincook4d_count
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d_fixedtgml
bash scripts/evaluate_train.sh checkpoints/do_captaincook4d_fixedtgml
bash scripts/evaluate_test.sh checkpoints/do_captaincook4d_fixedtgml
python3 validate.py --compare
```
