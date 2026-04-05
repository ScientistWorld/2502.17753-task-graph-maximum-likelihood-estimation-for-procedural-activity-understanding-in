# Scientist Constraints

## Model
The scientist must use the Direct Optimization (DO) model architecture: a learnable
scoring matrix that is row-normalized via softmax and masked to enforce DAG constraints
(no self-loops, beta→START, no edges to/from gamma). This is implemented in
`method/tgml.py::DirectOptimization`. The scientist may modify:
- The learning rate, beta parameter, and other hyperparameters
- The loss computation (e.g., alternative formulations of the TGML loss)
- The post-processing pipeline (threshold, transitive reduction)
- Initialization of the scoring matrix

The scientist may NOT:
- Replace the DO model with a neural network architecture (TGT or otherwise)
- Use pretrained weights or external models
- Change the overall formulation from direct adjacency matrix optimization

## Data
The scientist must use only the CaptainCook4D dataset from the cloned repository at
`method/repo/data/captaincook4d/`, which contains 24 cooking procedure scenarios
with ground truth task graphs and video annotations. No external datasets or
additional training data may be used.

## Evaluation
The scientist must evaluate using edge-level precision/recall/F1 against the ground
truth task graphs, matching the protocol in `method/train.py::train_do()`:
1. Extract core adjacency matrix (remove beta/gamma rows/columns)
2. Threshold at 1/n where n is the number of GT nodes
3. Add boundary edges (START→nodes with no incoming, END←nodes with no outgoing)
4. Apply transitive reduction
5. Compare with ground truth using sklearn metrics on the flattened matrices

## Training Procedure
The scientist may modify the training loop (optimizer choice, scheduling,
early stopping criteria) but must keep training at the scenario level
(one model per cooking procedure) and respect a maximum of 1000 epochs per scenario.

## Compliance Checklist
- [ ] Model is Direct Optimization (DO): learnable scoring matrix with softmax + DAG mask
- [ ] No pretrained models or external weights used
- [ ] Training data is CaptainCook4D only from the repo's data/captaincook4d directory
- [ ] Evaluation uses edge-level F1 on ground truth task graphs
- [ ] Post-processing includes thresholding, boundary edges, and transitive reduction
- [ ] Maximum 1000 epochs per scenario
