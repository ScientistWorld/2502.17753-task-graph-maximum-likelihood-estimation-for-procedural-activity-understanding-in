# Scientist Constraints

## Data
Use CaptainCook4D only for scored comparisons in this gym. No external datasets, extra human labels, or additional corpora should be introduced for the scored experiment.

## Evaluation
Use the existing evaluation contract:
1. produce one predicted graph per procedure,
2. score directed prerequisite edges against ground truth,
3. report precision, recall, and F1 in the `scores.json` structure written by `scripts/evaluate.sh`.

## Supervision Level
Keep the comparable weak-supervision setting: action sequences may be used for learning, but ground-truth task graphs are for evaluation only.

## Problem Scope
Stay within graph induction for procedural step dependencies. Do not change the task into a different prediction problem such as step classification, language modeling, or free-form instruction generation.

## Compliance Checklist
- [ ] CaptainCook4D only for the scored experiment
- [ ] No extra graph supervision during learning
- [ ] Output is one graph per procedure
- [ ] Evaluation uses the provided graph-edge scoring pipeline
- [ ] Results are written in the existing `scores.json` experiment format
