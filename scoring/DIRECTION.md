# Research Direction

## Research Problem
This paper addresses the problem of learning task graphs (directed acyclic graphs of
prerequisite relationships between steps) from observed action sequences in egocentric
videos. The core challenge is that step annotations are cheap, but full graph supervision
is expensive — so the method must infer graph structure from partial observations
(topological sorts of the graph).

## Core Contribution
The paper introduces the Task Graph Maximum Likelihood (TGML) loss, which treats the
task graph as a probabilistic model where the adjacency matrix defines conditional
probabilities between steps. The loss maximizes the likelihood of observed action
sequences under this model, with a contrastive term that penalizes edges from
unobserved nodes to observed nodes. The Direct Optimization (DO) model directly
optimizes an adjacency matrix using this loss.

## Approach Scope
The scientist should work on improving the task graph learning problem using the
TGML framework. They may:
- Modify the loss function formulation (different positive/contrastive terms,
  alternative graph likelihood models)
- Change the model architecture (different normalization, additional regularization,
  different DAG constraint formulations)
- Improve the training procedure (better optimization, curriculum, initialization)
- Tune hyperparameters (beta, threshold, learning rate)
- Improve post-processing (better transitive reduction, alternative graph cleanup)
- Design data augmentation strategies for sequences

## Out of Bounds
- Replacing the DO model with a neural network that uses video or text embeddings
  (TGT models). The paper's contribution is learning from action sequences alone,
  not multimodal fusion.
- Using external datasets beyond CaptainCook4D.
- Modifying the problem formulation (e.g., predicting node labels instead of edges).
- Ensemble methods combining multiple independently-trained models.
