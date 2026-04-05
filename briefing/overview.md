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
