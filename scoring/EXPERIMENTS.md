# Experiments

Both experiments score the same thing: how well a predicted prerequisite graph per procedure matches the ground-truth graph, measured on directed edges after the evaluator's fixed post-processing. The prediction contract, post-processing and metrics are in `briefing/evaluation.md`. Every method row in the score files is a directory under `checkpoints/`, and the evaluator adds its own `count_based` row.

## captaincook4d_task_graph (weight 0.7)

**What it establishes.** Whether a system can recover prerequisite relations between cooking steps from a few repeated executions of the same recipe (CaptainCook4D: 24 recipes, 7-25 key-steps, 1-13 error-free demonstrations each).

**Split.** 12 public procedures (`scores_train.json`, `scores.json`) and 12 held-out procedures (hidden test), balanced by graph size: 201 nodes and 240 edges on each side.

**Why it matters.** This is the main published benchmark for the problem. Demonstrations differ in valid step order, so the target is a partial order, not one sequence.

## noisy_demonstrations_task_graph (weight 0.3)

**What it establishes.** Whether the recovered structure survives imperfect demonstrations, as produced by real key-step recognisers or by people who skip and repeat steps.

**Data.** Procedure graphs have the size and shape of CaptainCook4D graphs: 7-25 key-steps, about 1.23 edges per node, mostly sequential with a few parallel branches. Each procedure has 4-10 demonstrations, sampled as random valid execution orders. Every key-step of a demonstration is then, with probability p = 0.05, replaced by another step, deleted, or preceded by an inserted random step, each equally likely. Repeated steps are dropped, keeping the first occurrence.

**Split.** 100 public procedures with graphs (`data/noisy_demos/train.json`) and 100 held-out procedures whose graphs are hidden (`data/noisy_demos/test_inputs.json` holds only their demonstrations).

**Why it matters.** Clean recordings make the observed order a faithful image of the graph. With noise, rules that trust every observed order break, and methods have to weigh evidence. The noise model is the insert/delete/replace perturbation used in the source paper's robustness study. The rate was fixed by a predeclared rule on public procedures: the hardest of {0.20, 0.10, 0.05} at which the reference stays at least 5 F1 above the Count-Based baseline.

## Scale notes

Published numbers (in `scoring/reference.json` for CaptainCook4D) average all 24 procedures; this gym reports the same metrics on fixed 12-procedure halves. No published numbers exist for the noisy-demonstration experiment.
