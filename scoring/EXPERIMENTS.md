# Experiments

This file describes the experiments in `scores.json` for future scientist agents.

## captaincook4d_task_graph

**What it establishes:** Whether a system can recover prerequisite relationships between cooking steps from repeated action sequences for the same recipe.

**Scientific purpose:** The benchmark is designed to reward methods that infer dependency structure rather than memorize a single canonical ordering. Multiple demonstrations of the same recipe can differ in valid step order, so the target is a graph, not one sequence.

**Current `scores.json` rows:** The reproduced score file currently contains `count_based` and `do`. These are just result labels. Future scientist agents may add their own method rows as long as they follow the same evaluation interface.

**How the experiment is scored:**
1. Predict one graph for each CaptainCook4D procedure.
2. Convert the prediction into a binary graph suitable for evaluation.
3. Compare predicted directed edges against ground-truth directed edges.
4. Average precision, recall, and F1 across procedures.

**Why it matters:** This is the paper's main graph-generation benchmark and the core gym target for this workspace.
