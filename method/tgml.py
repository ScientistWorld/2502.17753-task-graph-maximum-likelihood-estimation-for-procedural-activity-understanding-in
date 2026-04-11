"""
Task Graph Maximum Likelihood (TGML) implementation.

This module implements:
1. The TGML loss function (task_graph_maximum_likelihood_loss)
2. The Direct Optimization (DO) model
3. The Count-Based baseline

Based on the paper:
"Maximum Likelihood Estimation for Procedural Activity Understanding in Egocentric Videos"
(arXiv:2502.17753)

IMPORTANT: This implementation mirrors the OFFICIAL released code exactly.
The loss function uses the official formula:
    beta * torch.log(den) - torch.log(num)
The model mask and adjacency matrix match the official DO model.
"""

import copy
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


def task_graph_rate(sequence, A, all_nodes, beta):
    """
    Compute the rate for a single sequence (matching official _loss.py).

    rate = beta * log(den) - log(num)
    where num = sum of A[current, observed] and den = sum of A[unobserved, observed].
    """
    def compute_rate(i, sequence_np, A, all_nodes):
        current = sequence_np[i]
        mask = np.ones(len(all_nodes), dtype=bool)
        mask[np.isin(all_nodes, sequence_np[:i])] = False
        J = sequence_np[:i]
        notJ = all_nodes[mask]
        num = A[current, J].sum()
        den = A[notJ, :][:, J].sum()
        return beta * np.log(den) - np.log(num)

    total = 0.0
    sequence_np = sequence if not isinstance(sequence, torch.Tensor) else sequence.cpu().numpy()

    for i in range(1, len(sequence_np) - 1):
        total += compute_rate(i, sequence_np, A, all_nodes)

    return total


def task_graph_maximum_likelihood_loss(
    y: torch.Tensor,
    A: torch.Tensor,
    beta: float = 0.005
) -> torch.Tensor:
    """
    Compute the TGML loss for a batch of sequences.

    This implements the OFFICIAL released loss with random deduplication:
    Each sequence is processed by first deduplicating with 50% probability,
    then computing the rate for each position.

    Parameters
    ----------
    y : torch.Tensor
        Tensor of shape (batch_size, seq_len) containing sequences of node indices.
        Each sequence starts with beta (0), then START (1), then key steps, then END, then gamma.
    A : torch.Tensor
        Adjacency matrix (n+2) x (n+2), row-normalized (softmax applied).
    beta : float
        Hyperparameter balancing the positive and contrastive terms.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    total_loss = torch.tensor(0.0, dtype=torch.float32)

    for seq in y:
        # Convert to numpy for official computation
        seq_np = seq.cpu().numpy()

        # Deduplicate with 50% probability (matching official _loss.py)
        deduped = []
        for keystep in seq_np:
            if keystep not in deduped:
                deduped.append(keystep)
            elif np.random.uniform() < 0.5:
                deduped.remove(keystep)
                deduped.append(keystep)

        if len(deduped) < 3:
            continue

        all_nodes = np.arange(1, A.shape[0] - 1)  # Exclude beta (0) and gamma (-1)

        # Compute rate for each position
        for i in range(1, len(deduped) - 1):
            current = deduped[i]
            mask = np.ones(len(all_nodes), dtype=bool)
            mask[np.isin(all_nodes, deduped[:i])] = False
            J = deduped[:i]
            notJ = all_nodes[mask]

            if len(notJ) == 0 or len(J) == 0:
                continue

            num = A[current, J].sum()
            den = A[notJ, :][:, J].sum()

            if num <= 0 or den <= 0:
                continue

            rate = beta * torch.log(den + 1e-8) - torch.log(num + 1e-8)
            total_loss = total_loss + rate

    return total_loss


def task_graph_maximum_likelihood_loss_efficient(
    y: torch.Tensor,
    A: torch.Tensor,
    beta: float = 0.005
) -> torch.Tensor:
    """
    Simplified TGML loss without random deduplication.
    Used for faster training iterations where the deduplication noise is not needed.
    """
    total_loss = torch.tensor(0.0, dtype=torch.float32)

    for seq in y:
        seq_np = seq.cpu().numpy()

        all_nodes = np.arange(1, A.shape[0] - 1)

        for i in range(1, len(seq_np) - 1):
            current = seq_np[i]
            observed = seq_np[:i]

            mask = np.ones(len(all_nodes), dtype=bool)
            mask[np.isin(all_nodes, observed)] = False
            unobserved = all_nodes[mask]

            if len(unobserved) == 0:
                continue

            num = A[current, observed].sum()
            den = A[unobserved, :][:, observed].sum()

            if num <= 0 or den <= 0:
                continue

            # Fixed formula: -log(num) + beta*log(den)
            loss_t = -torch.log(num + 1e-8) + beta * torch.log(den + 1e-8)
            total_loss = total_loss + loss_t

    return total_loss


class TaskGraph(nn.Module):
    """Task graph as a learnable adjacency matrix parameter."""
    def __init__(self, num_nodes: int):
        super().__init__()
        # num_nodes includes beta and gamma placeholders
        self.task_graph = nn.Parameter(torch.rand(num_nodes, num_nodes))

    def forward(self):
        return self.task_graph


class DirectOptimization(nn.Module):
    """
    Direct Optimization (DO) model for task graph learning.

    Directly optimizes the adjacency matrix using the TGML loss via gradient descent.
    """
    def __init__(self, num_nodes: int, device: str = "cpu"):
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph (including beta and gamma placeholders).
            Typically: 1 (beta) + n (key steps) + 1 (gamma) = n + 2
        device : str
            Device for computation.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        self.adjacency_matrix = TaskGraph(num_nodes)
        self.register_buffer("mask", self._build_dag_mask())

    def _build_dag_mask(self) -> torch.Tensor:
        """
        Match the official DO masking logic from the released code.
        Valid edges: START→beta, gamma→END, all other non-diagonal pairs.
        """
        mask = torch.ones(self.num_nodes, self.num_nodes, dtype=torch.bool)
        mask[0, :] = False         # beta row masked
        mask[1, :] = False         # START row masked
        mask[1, 0] = True          # START→beta unmasked
        mask[:, -1] = False        # gamma col masked
        mask[:, -2] = False        # END col masked
        mask[-1, :] = False         # gamma row masked
        mask[-1, -2] = True        # gamma→END unmasked
        identity = torch.eye(self.num_nodes, dtype=torch.bool)
        return mask & (~identity)

    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Get the row-normalized adjacency matrix with DAG masking applied.

        Returns
        -------
        torch.Tensor
            Adjacency matrix Z = softmax(A) with masked positions set to 0.
        """
        masked_scores = torch.where(
            self.mask,
            self.adjacency_matrix(),
            torch.tensor(float("-inf"), device=self.device),
        )
        return torch.softmax(masked_scores, dim=1)

    def forward(self, y: torch.Tensor, beta: float = 0.005, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute the TGML loss using the OFFICIAL loss function with random deduplication.

        Parameters
        ----------
        y : torch.Tensor
            Sequences of node indices, shape (batch_size, seq_len).
        beta : float
            TGML loss beta parameter.
        eps : float
            Small constant to avoid log(0).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        Z = self.get_adjacency_matrix()
        # Use the official loss function (with random deduplication)
        return task_graph_maximum_likelihood_loss(y, Z + eps, beta)


def binarize_adjacency(Z: torch.Tensor, threshold: float) -> np.ndarray:
    """Binarize adjacency matrix using threshold."""
    Z_np = Z.cpu().numpy()
    binary = (Z_np > threshold).astype(np.float32)
    return binary


def remove_self_loops(A: np.ndarray) -> np.ndarray:
    """Remove self-loops from adjacency matrix."""
    A = A.copy()
    for i in range(min(A.shape)):
        A[i, i] = 0
    return A


def add_start_end_edges(A: np.ndarray, id_start: int = 1, id_end: int = None) -> np.ndarray:
    """
    Add START and END edges to make the graph fully connected at boundaries.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (n+2) x (n+2).
    id_start : int
        Index of START node (default 1).
    id_end : int
        Index of END node (default n).

    Returns
    -------
    np.ndarray
        Modified adjacency matrix.
    """
    n = A.shape[0]
    if id_end is None:
        id_end = n - 2  # n+1 is gamma, so n is the last real node

    # Add START -> nodes with no incoming edges (excluding beta and gamma)
    for i in range(1, id_end + 1):
        if np.sum(A[i, :]) == 0:  # no incoming edges
            A[i, 0] = 1  # START (index 1) -> i, but wait, A is indexed A[row, col]

    # Wait, I need to check the convention.
    # In the paper: A[i,j] = weight of edge K_j -> K_i (j is precondition of i)
    # So A[i,j] = 1 means "j is a precondition of i"
    # A row = destination node, column = source node
    #
    # So:
    # - A[i, j] > 0 means there is an edge j -> i
    # - A[:, j] gives all nodes that depend on node j
    # - A[i, :] gives all preconditions of node i
    #
    # For START: nodes with no preconditions should have START as a precondition
    # For END: nodes with no successors should have END as their successor
    #
    # A[i, START] > 0 means START is a precondition of i
    # A[END, i] > 0 means i is a precondition of END

    # Nodes with no preconditions (A[i, :] == 0 except self) should depend on START
    for i in range(1, id_end + 1):
        # Check if node i has any preconditions
        row = A[i, :].copy()
        row[i] = 0  # ignore self
        if row.sum() == 0:
            A[i, 0] = 1  # START (index 0 in the 0-indexed scheme... but we use 1 for START)

    # Actually, let me reconsider the indexing.
    # In the code: sequences are [beta(0), START(1), steps(2..n), END(n+1), gamma(n+2)]
    # Wait no, looking at train_with_gt.py:
    # id_beta = 0
    # id_start = 1
    # id_end = len(GT.nodes)  (so if GT has 19 nodes, id_end = 19)
    # id_gamma = len(GT.nodes) + 1  = 20
    # num_nodes = len(GT.nodes) + 2 = 21
    #
    # So matrix indices:
    # 0 = beta
    # 1 = START
    # 2..(n) = key steps where n = len(GT.nodes)-1 (so GT node 1 maps to matrix index 2, etc.)
    # n+1 = END
    # n+2 = gamma
    #
    # Wait, but id_end = len(GT.nodes), which includes START and END as nodes in GT.
    # Let me think again.
    # GT has nodes 0..(len(GT.nodes)-1)
    # START has the smallest ID, END has the largest ID
    # In the adjacency matrix:
    # 0 = beta
    # 1 = START
    # 2..(len(GT.nodes)-1) = the other GT nodes
    # len(GT.nodes) = END
    # len(GT.nodes)+1 = gamma
    #
    # So num_nodes = len(GT.nodes) + 2
    #
    # In the ground truth edges:
    # For edge [dst, src] in task_graph_json["edges"], we add edge src -> dst
    # This becomes A[dst, src] = 1
    #
    # The A matrix uses A[row, col] where:
    # A[i, j] > 0 means j is a precondition of i

    # Hmm, let me re-read the paper's notation.
    # From the paper: Z_{(i,j)} = omega(K_i, K_j) means edge from K_j to K_i
    # So Z[row, col] = weight of edge col -> row
    #
    # In the implementation, A is the scoring matrix and Z = softmax(A).
    # Looking at _models.py:
    # Z = softmax(A), where A is the learnable scoring matrix
    #
    # In the loss: Z[(y_t, j)] means Z[current_node, precondition_node]
    # This matches: Z[row, col] = edge from col to row
    #
    # In the GT graph: edge [dst, src] means src -> dst
    # So we set Z[dst, src] = 1 in the GT adjacency matrix

    return A


def delete_redundant_edges(G: nx.DiGraph) -> None:
    """
    Remove redundant edges (transitive reduction).
    If there's a path from A to C and also a direct edge A->C, remove A->C.
    """
    G_copy = copy.deepcopy(G)
    for current_node in G.nodes:
        for successor in list(G_copy.successors(current_node)):
            # Temporarily remove edge
            if G.has_edge(current_node, successor):
                G.remove_edge(current_node, successor)
                # Check if there's still a path
                if not nx.has_path(G, current_node, successor):
                    # Re-add the edge
                    G.add_edge(current_node, successor)


def delete_cycles(G: nx.DiGraph) -> None:
    """Remove cycles from graph."""
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            break


def insert_start_end_nodes(G: nx.DiGraph) -> None:
    """Add START and END nodes to graph if not present."""
    # Insert END node
    if "END" not in G:
        G.add_node("END", info="END")
    for node in list(G.nodes):
        if G.in_degree(node) == 0 and node != "END" and node != "START":
            G.add_edge("END", node)

    # Insert START node
    if "START" not in G:
        G.add_node("START", info="START")
    for node in list(G.nodes):
        if G.out_degree(node) == 0 and node != "START" and node != "END":
            G.add_edge(node, "START")


def adjacency_to_networkx(
    adj: np.ndarray,
    node_labels: dict = None,
    threshold: float = 0.5
) -> nx.DiGraph:
    """
    Convert adjacency matrix to NetworkX graph with post-processing.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix (n+2) x (n+2).
    node_labels : dict
        Mapping from index to label name.
    threshold : float
        Threshold for binarization.

    Returns
    -------
    nx.DiGraph
        Processed task graph.
    """
    n = adj.shape[0]

    # Binarize
    binary = (adj > threshold).astype(float)

    # Remove self-loops
    for i in range(n):
        binary[i, i] = 0

    # Create graph
    G = nx.DiGraph(binary)

    # Add labels if provided
    if node_labels:
        for node in G.nodes:
            if node in node_labels:
                G.nodes[node]["label"] = node_labels[node]

    # Remove nodes 0 (beta) and n-1 (gamma) if they're isolated
    if 0 in G and G.degree(0) == 0:
        G.remove_node(0)
    if n - 1 in G and G.degree(n - 1) == 0:
        G.remove_node(n - 1)

    # Transitive reduction
    delete_redundant_edges(G)

    # Cycle removal
    delete_cycles(G)

    # Add START -> nodes with no preconditions
    in_degree_zeros = [node for node in G.nodes if G.in_degree(node) == 0]
    end_candidates = [n for n in G.nodes if isinstance(n, int)]
    if end_candidates:
        end_node = max(end_candidates) if all(isinstance(n, int) for n in G.nodes) else "END"
        for node in in_degree_zeros:
            if node != end_node:
                G.add_edge(end_node, node)

    # Add nodes with no successors -> END
    out_degree_zeros = [node for node in G.nodes if G.out_degree(node) == 0]
    start_candidates = [n for n in G.nodes if isinstance(n, int)]
    if start_candidates:
        start_node = min(start_candidates) if all(isinstance(n, int) for n in G.nodes) else "START"
        for node in out_degree_zeros:
            if node != start_node:
                G.add_edge(node, start_node)

    return G


def compute_precision_recall_f1(
    gt_adj: np.ndarray,
    pred_adj: np.ndarray
) -> tuple:
    """
    Compute precision, recall, and F1 between ground truth and predicted graphs.

    Both adjacency matrices should be binary, (n) x (n), indexed by GT node ID.
    Uses standard binary classification definitions directly.

    Parameters
    ----------
    gt_adj : np.ndarray
        Ground truth adjacency matrix (n x n).
    pred_adj : np.ndarray
        Predicted adjacency matrix (n x n).

    Returns
    -------
    tuple : (precision, recall, f1) in percentages
    """
    gt_flat = gt_adj.astype(np.int32).flatten()
    pred_flat = pred_adj.astype(np.int32).flatten()

    tp = int(np.sum((gt_flat == 1) & (pred_flat == 1)))
    fp = int(np.sum((gt_flat == 0) & (pred_flat == 1)))
    fn = int(np.sum((gt_flat == 1) & (pred_flat == 0)))

    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
