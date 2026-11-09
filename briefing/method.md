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
- The adjacency matrix is $Z = \text{softmax}(A)$, with masking to enforce DAG constraints (masking diagonal, START row, END column)
- Trained with Adam optimizer (lr=0.1), $\beta=0.005$, max 1000 epochs. $\beta$ is raised to 0.5 for a procedure when none of its sequences contains every key-step (Appendix A-C).
- Early stopping on the weakly supervised Sequence Accuracy (SA, Appendix A-B). After every epoch, $Z$ is binarised at $1/n$, completed with START/END edges and transitively reduced. If the result is a DAG, SA measures how many of each step's predicted predecessors appear before it in the training sequences. Once SA > 0.95, training halts after 25 epochs without SA improvement (a repeated SA of 1.0 counts as no improvement), and the state at the halting epoch is the output.
- One model per procedure; results are averaged over 5 runs (seeds).

In this gym the reference implementation is `method/train.py` (`bash scripts/method.sh` writes `checkpoints/do/`, 5 seeds × all procedures of both experiments, CPU, about 15 minutes).

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
