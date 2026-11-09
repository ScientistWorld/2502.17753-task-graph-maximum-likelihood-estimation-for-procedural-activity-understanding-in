# Task Graph Maximum Likelihood Estimation for Procedural Activity Understanding in Egocentric Videos

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening).

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2502.17753"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2502.17753-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2502.17753](https://arxiv.org/abs/2502.17753) |
|:---|:---|
| **Domain** | Computer Vision / Video Understanding / Graph Learning |
| **TL;DR** | A differentiable maximum likelihood loss for learning task graphs (DAGs of action ordering constraints) from video demonstrations of procedural activities. |

## Summary

The paper addresses the problem of automatically learning task graphs — Directed Acyclic Graphs encoding which procedural steps depend on which other steps — from observations of people performing activities (e.g., cooking). The key contribution is the **Task Graph Maximum Likelihood (TGML)** loss, which frames task graph learning as maximum likelihood estimation: the loss maximizes the probability of observing the training action sequences under the graph's ordering constraints. The paper proposes two models: **Direct Optimization (DO)** which directly optimizes the adjacency matrix via gradient descent, and **Task Graph Transformer (TGT)** which predicts graphs from video or text embeddings using a neural architecture. Evaluated on CaptainCook4D, EgoPER, and EgoProceL datasets for task graph generation, and on Ego-Exo4D and Assembly101-O/EPIC-Tent-O for downstream tasks.

## Problem

The task is to infer a **task graph** for a procedural activity from repeated demonstrations of that activity. A task graph is a directed acyclic graph whose nodes are procedure steps and whose edges represent prerequisite relationships between steps.

The available supervision is weak: for each procedure, the learner sees only action sequences collected from multiple demonstrations. It must recover the hidden prerequisite structure without being trained directly on graph labels.

## The paper's method

The paper introduces **Task Graph Maximum Likelihood (TGML)**, a novel differentiable loss function that enables learning task graphs from action sequences via maximum likelihood estimation. The key insight is framing task graph learning as estimating the adjacency matrix that maximizes the likelihood of observing the training sequences under the graph's ordering constraints.

## Key results of the paper

- **DO achieves F1 of 87.8% on CaptainCook4D**, 91.0% on EgoPER, and 72.3% on EgoProceL for task graph generation — improvements of +14.5%, +10.2%, and +13.6% over the best prior methods respectively
- **For online mistake detection**, DO achieves +19.8% and +6.4% average F1 improvement over PREGO on Assembly101-O and EPIC-Tent-O respectively
- **On Ego-Exo4D procedure understanding**, TGT and DO achieve top performance across 5 downstream tasks (previous keysteps, optional keysteps, procedural mistakes, missing keysteps, future keysteps)

## Run it

```bash
pip install sciworld
sw research 2502.17753 --mode innovate --agent claude-code
sw evaluate 2502.17753
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

