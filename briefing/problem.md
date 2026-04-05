# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

## Research Question

The paper tackles the problem of **learning task graphs from demonstrations** of procedural activities. Procedural activities (e.g., cooking a recipe, assembling an object) consist of sequences of key-steps with ordering constraints between them. The goal is to automatically infer these ordering constraints — expressed as a Directed Acyclic Graph (DAG) where nodes are key-steps and edges encode preconditions — from observed action sequences in videos or text.

Given a set of example action sequences (e.g., multiple videos of someone making coffee, each showing the steps in slightly different orders), the problem is to recover the underlying task graph that captures which steps depend on which other steps. This is a weakly supervised problem: the model observes action sequences but does not have direct graph supervision.

## Why It Matters

Task graphs provide an explicit, human-readable representation of procedures that captures the partial ordering of steps. They can support downstream video understanding tasks such as:
- Detecting procedural mistakes (a step performed out of order)
- Predicting missing or future steps
- Recognizing key-steps in video

Current approaches rely on hand-crafted graph mining procedures. This paper proposes a learning-based approach with a differentiable objective, enabling end-to-end optimization.

## Success Criteria

A successful solution should:
1. **Recover accurate task graphs** from action sequences — predicted graphs should match ground truth preconditions with high precision, recall, and F1 score
2. **Generalize across datasets** — the same method should work on different procedural domains (cooking, assembly, etc.)
3. **Support downstream tasks** — learned graphs should be useful for mistake detection and other video understanding tasks
4. **Be differentiable** — the objective should be optimizable via gradient descent, enabling integration into neural networks
