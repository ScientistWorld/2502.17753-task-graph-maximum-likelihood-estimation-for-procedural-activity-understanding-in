"""
Evaluation metrics for task graph generation.

Contains:
- compute_precision_recall_f1: Edge-level P/R/F1
- delete_redundant_edges: Transitive reduction
- delete_cycles: Cycle removal
- post_process_adjacency: Full post-processing pipeline matching the paper's approach

This module is evaluation-only and has NO imports from method/.
"""

import numpy as np
import networkx as nx
import copy


def compute_precision_recall_f1(
    gt_adj: np.ndarray,
    pred_adj: np.ndarray
) -> tuple:
    """
    Compute precision, recall, and F1 between ground truth and predicted graphs.

    Both adjacency matrices should be binary, (n) x (n), indexed by GT node ID.
    Uses the standard binary definitions directly to avoid an unnecessary runtime
    dependency in the reusable evaluator.

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
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def delete_redundant_edges(G: nx.DiGraph) -> None:
    """
    Remove redundant edges (transitive reduction).
    If there's a path from A to C and also a direct edge A->C, remove A->C.
    """
    G_copy = copy.deepcopy(G)
    for current_node in G.nodes:
        for successor in list(G_copy.successors(current_node)):
            if G.has_edge(current_node, successor):
                G.remove_edge(current_node, successor)
                if not nx.has_path(G, current_node, successor):
                    G.add_edge(current_node, successor)


def delete_cycles(G: nx.DiGraph) -> None:
    """Remove cycles from graph."""
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            break


def add_boundary_edges(G: nx.DiGraph, id_start: int, id_end: int) -> None:
    """Match the official boundary-edge insertion used by DO."""
    in_degree_zeros = [
        node for node in G.nodes
        if G.in_degree(node) == 0 and node != id_end and node != id_start
    ]
    for node in in_degree_zeros:
        G.add_edge(id_end, node)

    out_degree_zeros = [
        node for node in G.nodes
        if G.out_degree(node) == 0 and node != id_end and node != id_start
    ]
    for node in out_degree_zeros:
        G.add_edge(node, id_start)


def post_process_adjacency(
    pred_core: np.ndarray,
    n_gt: int
) -> np.ndarray:
    """
    Full post-processing pipeline matching the paper's evaluation approach.

    Steps:
    1. Binarize at threshold 1/n
    2. Add boundary edges (START→isolated nodes, isolated nodes→END)
    3. Transitive reduction
    4. Cycle removal

    Parameters
    ----------
    pred_core : np.ndarray
        Core adjacency matrix (n_gt x n_gt), raw scores.
    n_gt : int
        Number of GT nodes.

    Returns
    -------
    np.ndarray
        Processed adjacency matrix (n_gt x n_gt).
    """
    # Threshold matches official code: 1/(num_nodes - 2) where num_nodes = n_gt + 2
    threshold = 1.0 / (n_gt + 2)

    # Binarize matching official: np.where(pred_core < threshold, 0, 1)
    pred_binary = np.where(pred_core < threshold, 0, 1).astype(float)

    # Build graph matching official: pred_binary[row, col] = 1 means edge col -> row
    GP = nx.DiGraph(pred_binary)

    # Add boundary edges
    id_start = 0  # GT node ID
    id_end = n_gt - 1  # GT node ID

    add_boundary_edges(GP, id_start=id_start, id_end=id_end)

    # Transitive reduction (no cycle removal -- official train_with_gt.py doesn't use it)
    delete_redundant_edges(GP)

    # pred_adj has graph orientation: adj[u,v]=1 means edge u->v (u is precondition of v)
    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_adj[u, v] = 1.0

    return pred_adj
