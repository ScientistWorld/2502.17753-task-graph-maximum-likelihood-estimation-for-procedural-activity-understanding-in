# Experiments

This file describes the experiments in `scores.json` for future scientist agents.

## captaincook4d_task_graph

**What it establishes:** Tests whether the method can recover accurate task graphs (directed acyclic graphs of prerequisite relationships between procedural steps) from observed action sequences in cooking videos. This is the paper's primary benchmark.

**Scientific purpose:** Given multiple example action sequences from the same cooking procedure (e.g., multiple videos of someone making coffee), the system must infer which steps depend on which other steps — without direct graph supervision. The predicted graphs are evaluated against ground-truth task graphs annotated by humans.

**What each method in the experiment represents:**
- `count_based`: A traditional graph-mining baseline that counts co-occurrences and transition probabilities in the action sequences.
- `do`: The paper's proposed Direct Optimization method, which directly optimizes an adjacency matrix using a differentiable maximum likelihood loss.
- Other baselines (msgi, llama3, msg2) are additional comparison points from the paper.

**How the experiment works:**
1. For each of 24 cooking procedures, learn a task graph from the procedure's action sequences.
2. Post-process the predicted graph (binarize, add boundary edges, transitive reduction).
3. Compare predicted graph edges against ground-truth edges using precision, recall, and F1.
4. Average scores across all 24 procedures.

**Primary claim validated:** The DO method achieves substantially higher F1 than the count-based baseline on task graph recovery.
