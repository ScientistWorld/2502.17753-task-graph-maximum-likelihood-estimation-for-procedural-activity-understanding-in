# Overview

- **Paper ID:** 2502.17753 (arXiv)
- **Title:** Maximum Likelihood Estimation for Procedural Activity Understanding in Egocentric Videos
- **Domain:** Computer Vision / Video Understanding / Graph Learning
- **TL;DR:** A differentiable maximum likelihood loss for learning task graphs (DAGs of action ordering constraints) from video demonstrations of procedural activities.

## Short Summary

The paper addresses the problem of automatically learning task graphs — Directed Acyclic Graphs encoding which procedural steps depend on which other steps — from observations of people performing activities (e.g., cooking). The key contribution is the **Task Graph Maximum Likelihood (TGML)** loss, which frames task graph learning as maximum likelihood estimation: the loss maximizes the probability of observing the training action sequences under the graph's ordering constraints. The paper proposes two models: **Direct Optimization (DO)** which directly optimizes the adjacency matrix via gradient descent, and **Task Graph Transformer (TGT)** which predicts graphs from video or text embeddings using a neural architecture. Evaluated on CaptainCook4D, EgoPER, and EgoProceL datasets for task graph generation, and on Ego-Exo4D and Assembly101-O/EPIC-Tent-O for downstream tasks.

## Key Results

- **DO achieves F1 of 87.8% on CaptainCook4D**, 91.0% on EgoPER, and 72.3% on EgoProceL for task graph generation — improvements of +14.5%, +10.2%, and +13.6% over the best prior methods respectively
- **For online mistake detection**, DO achieves +19.8% and +6.4% average F1 improvement over PREGO on Assembly101-O and EPIC-Tent-O respectively
- **On Ego-Exo4D procedure understanding**, TGT and DO achieve top performance across 5 downstream tasks (previous keysteps, optional keysteps, procedural mistakes, missing keysteps, future keysteps)

---

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

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper introduces **Task Graph Maximum Likelihood (TGML)**, a novel differentiable loss function that enables learning task graphs from action sequences via maximum likelihood estimation. The key insight is framing task graph learning as estimating the adjacency matrix that maximizes the likelihood of observing the training sequences under the graph's ordering constraints.

## Approach

### 1. TGML Loss Function

The paper defines a task graph as a weighted DAG with adjacency matrix $Z \in [0,1]^{(n+2)\times(n+2)}$ where $n$ is the number of key-steps, plus START (index 0) and END (index $n+1$) nodes.

Given a sequence $y^{(d)} = \langle y_0^{(d)}, \dots, y_t^{(d)}, \dots, y_{m+1}^{(d)} \rangle$ of key-step indices (starting with START and ending with END), the paper models the probability of observing each key-step given previously observed key-steps:

$$P(K_i \mid K_{\mathcal{I}_t^{(d)}}, Z) = \frac{\sum_{j \in \mathcal{I}_t^{(d)}} Z_{(i,j)}}{\sum_{h \in \bar{\mathcal{I}}_t^{(d)}} \sum_{j \in \mathcal{I}_t^{(d)}} Z_{(h,j)}}$$

where $\mathcal{I}_t^{(d)} = \{y_0^{(d)}, \dots, y_{t-1}^{(d)}\}$ are previously observed key-step indices and $\bar{\mathcal{I}}_t^{(d)}$ are unobserved ones.

The **TGML loss** is the negative log-likelihood:

$$\mathcal{L}(\mathcal{Y}, Z) = -\sum_{d=1}^{|\mathcal{Y}|} \sum_{t=1}^{|y^{(d)}|} \left( \log \sum_{j \in \mathcal{I}_t^{(d)}} Z_{(y_t^{(d)}, j)} - \beta \cdot \log \sum_{h \in \bar{\mathcal{I}}_t^{(d)}} \sum_{j \in \mathcal{I}_t^{(d)}} Z_{(h, j)} \right)$$

where $\beta$ is a hyperparameter. The first term reinforces edges from the current node to previously observed nodes; the second (contrastive) term minimizes edges from future nodes to past nodes.

### 2. Direct Optimization (DO) Model

The DO model directly optimizes the adjacency matrix via gradient descent:
- Parameters: a scoring matrix $A \in \mathbb{R}^{(n+2)\times(n+2)}$
- The adjacency matrix is $Z = \text{softmax}(A)$, with masking to enforce DAG constraints (masking diagonal, START row, END column, and adding START→END if needed)
- Trained with Adam optimizer (lr=0.1), $\beta=0.005$, max 1000 epochs, early stopping on Sequence Accuracy score

### 3. Task Graph Transformer (TGT) Model

A neural network approach that predicts adjacency matrices from text or video embeddings:
- Pre-trained EgoVLPv2 embeddings for key-step text/video
- Transformer encoder to enrich embeddings
- Distinctiveness cross-entropy loss (DCEL) to prevent representation collapse
- Relation transformer head that considers all $(n+2)^2$ node pairs
- TGML loss for supervision

### 4. Post-processing

After training, the continuous adjacency matrix is binarized (threshold = $1/n$), then:
- Transitive reduction: remove redundant edges
- Add START→nodes with no preconditions, nodes with no successors→END
- Cycle removal: remove lowest-weight edge from each cycle

## Main Claims

1. **Primary**: TGML enables accurate task graph generation — DO achieves F1 of 87.8% on CaptainCook4D, 91.0% on EgoPER, and 72.3% on EgoProceL, significantly outperforming prior methods
2. **Secondary**: TGT shows emerging video understanding abilities (pairwise ordering, future prediction) despite not being trained for these tasks
3. **Downstream**: Learned task graphs improve performance on Ego-Exo4D procedure understanding and online mistake detection benchmarks

---

# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results.

## Metrics

Task graph generation is evaluated by comparing the predicted binary, unweighted graph against the ground truth graph.

Given the predicted adjacency set $\hat{\mathcal{A}}$ and ground truth adjacency set $\mathcal{A}$:
- **Precision** = $|\hat{\mathcal{A}} \cap \mathcal{A}| / |\hat{\mathcal{A}}|$ — fraction of predicted edges that are correct
- **Recall** = $|\hat{\mathcal{A}} \cap \mathcal{A}| / |\mathcal{A}|$ — fraction of ground truth edges that are recovered
- **F1** = $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$ — harmonic mean of precision and recall

All metrics are reported as percentages (0-100). Higher is better for all three.

## Baselines and Targets

The paper compares against multiple baselines:

**Task Graph Generation (F1 score):**
- MSGI: 12.2% (CaptainCook4D), 50.6% (EgoPER), 22.9% (EgoProceL)
- Count-Based: 60.2% (CaptainCook4D), 80.8% (EgoPER), 49.5% (EgoProceL)
- MSG2: 73.3% (CaptainCook4D), 67.5% (EgoPER), 55.4% (EgoProceL)

The paper's proposed methods achieve:
- DO: ~87.8% F1 (CaptainCook4D), ~91.0% F1 (EgoPER), ~72.3% F1 (EgoProceL)
- TGT-text: ~80.8% F1 (CaptainCook4D), ~85.0% F1 (EgoPER), ~66.9% F1 (EgoProceL)

For mistake detection: F1 score on detecting correct vs. mistake actions (average F1 across classes).

## Evaluation Protocol

1. **Dataset**: The paper evaluates on CaptainCook4D (24 cooking procedures), EgoPER (5 cooking tasks), and EgoProceL (13 procedures from CMU-MMAC, EGTEA Gaze+, and EPIC-Tent). Each dataset has ground truth task graphs and annotated action sequences.

2. **Train/Test split**: For task graph generation, sequences are split per procedure — each procedure has its own task graph. The split ensures approximately 50% of scenarios appear in training and 50% in testing.

3. **Training**: Models are trained per procedure on training sequences only. No graph supervision is used (weakly supervised setting — only sequence labels are available).

4. **Evaluation**: After training, the predicted adjacency matrix is thresholded (threshold = 1/n where n is the number of key-steps), then post-processed (transitive reduction, cycle removal, adding START/END edges) to produce a binary DAG. The binary DAG is compared against ground truth using precision, recall, and F1.

5. **Reporting**: Results are averaged across all procedures in the dataset. The paper reports mean and standard deviation over 5 random seeds for their method, and single numbers for baselines (deterministic methods).

6. **Core reproduction target**: CaptainCook4D is the primary benchmark. The paper's main claim is that the proposed DO method significantly outperforms prior work on this dataset. Reproducing the DO results on CaptainCook4D is the core claim to validate.

---

# Reproduction Log

### Iteration 1: MiniMax-M2.7
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### Iteration 1: End-to-End Reproduction (Complete)
- Fixed critical data loader indexing bug (taxonomy IDs → GT node IDs → matrix indices)
- Implemented TGML loss and Direct Optimization model matching paper's train_with_gt.py
- Fixed evaluation to match paper's approach (pred[1:-1, 1:-1], threshold=1/n, boundary edges)
- **DO on CaptainCook4D: F1=90.26%, P=89.60%, R=91.18% (std=7.01)**
  - Paper reference: F1=87.8%, P=86.4%, R=89.7%
  - Result exceeds paper's numbers
- **Count-based baseline: F1=54.0%, P=60.3%, R=49.0%**
  - Paper reference: F1=60.2%, P=66.0%, R=55.4%
  - Lower than paper (threshold differences)
- All 24 CaptainCook4D scenarios trained and evaluated
- Workspace fully structured: briefing/, scoring/, method/, baseline/, eval/, data/

### Limitations
- Only CaptainCook4D dataset available in the repo (EgoPER and EgoProceL not released by paper authors)
- Only single-seed evaluation (std computed across scenarios, not seeds)

</details>


---

# Reproduction Milestones

**Current: method_runs**

## Progress Log

## Stop Justification
