# Scientist Constraints

## Data
- Learn only from the demonstrations provided for each procedure: CaptainCook4D annotations and `data/noisy_demos/`.
- Do not use external datasets, extra human labels, text corpora, language models or step-name knowledge to obtain graphs.
- Never use ground-truth graphs of held-out procedures from any source, including re-downloading the original CaptainCook4D graph files.
- For a noisy-demonstration procedure, use only its own (noisy) demonstrations.

## Supervision
- Weak supervision: demonstrations are the learning signal.
- Public graphs (12 CaptainCook4D procedures, 100 noisy-benchmark procedures) may be used for development and model selection, e.g. choosing hyperparameters, but not as training labels for held-out procedures.

## Output
- One graph per procedure, written as `checkpoints/<method>/<procedure>_predictions.npz` with `pred_cores` of shape (5, n, n). Values are finite and in [0, 1]; the five entries are independent trials.
- Every procedure of every experiment must be covered. Partial coverage fails the experiment.
- Scores come only from `scripts/evaluate.sh`, `scripts/evaluate_train.sh` and the hidden `scripts/evaluate_test.sh`. Do not edit `eval/` or the evaluators' data, and do not hand-write score files.

## Problem scope
Stay within graph induction for procedural step dependencies. The evaluator's threshold (1/n) and post-processing are fixed; a method controls only the matrices it writes.

## Compliance checklist
- [ ] Only the provided demonstrations used to infer each graph
- [ ] No held-out graphs from any source
- [ ] Five trials per procedure, values finite in [0, 1], all procedures covered
- [ ] Scores produced by the provided evaluators
