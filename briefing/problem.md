# Problem Definition

## Research Question

The task is to infer a **task graph** for a procedural activity from repeated demonstrations of that activity. A task graph is a directed acyclic graph whose nodes are procedure steps and whose edges represent prerequisite relationships between steps.

The available supervision is weak: for each procedure, the learner sees only action sequences collected from multiple demonstrations. It must recover the hidden prerequisite structure without being trained directly on graph labels.

## Why It Matters

Task graphs provide an explicit, human-readable representation of procedures that captures the partial ordering of steps. They can support downstream video understanding tasks such as:
- Detecting procedural mistakes (a step performed out of order)
- Predicting missing or future steps
- Recognizing key-steps in video

The challenge is that real procedures permit multiple valid orders, so a good system must recover dependency structure rather than memorize one dominant sequence.

## Success Criteria

A successful solution should:
1. **Recover accurate task graphs** from action sequences, measured by agreement with held-out ground-truth prerequisite edges.
2. **Handle procedures with multiple valid execution orders**, not just one fixed sequence.
3. **Generalize across procedures within a benchmark**, since evaluation averages performance over many different recipes/tasks.
4. **Preserve the weak-supervision setting**, where action sequences are available for learning but graph annotations are reserved for evaluation.
