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
    Uses sklearn metrics for direct comparison (matching the paper's evaluation).

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
    from sklearn.metrics import precision_score, recall_score, f1_score

    gt_flat = gt_adj.flatten()
    pred_flat = pred_adj.flatten()

    precision = precision_score(gt_flat, pred_flat, zero_division=0) * 100
    recall = recall_score(gt_flat, pred_flat, zero_division=0) * 100
    f1 = f1_score(gt_flat, pred_flat, zero_division=0) * 100

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
    threshold = 1.0 / n_gt

    # Binarize
    pred_binary = (pred_core > threshold).astype(float)

    # Build NetworkX graph
    GP = nx.DiGraph(pred_binary)

    # Add boundary edges
    id_start = 0  # GT node ID
    id_end = n_gt - 1  # GT node ID

    # Nodes with no incoming edges (except END) → connect from END
    in_degree_zeros = [n for n in GP.nodes if GP.in_degree(n) == 0 and n != id_end]
    for node in in_degree_zeros:
        GP.add_edge(id_end, node)

    # Nodes with no outgoing edges (except START) → connect to START
    out_degree_zeros = [n for n in GP.nodes if GP.out_degree(n) == 0 and n != id_start]
    for node in out_degree_zeros:
        GP.add_edge(node, id_start)

    # Transitive reduction
    delete_redundant_edges(GP)

    # Cycle removal
    delete_cycles(GP)

    # Convert back to adjacency matrix
    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_adj[v, u] = 1.0

    return pred_adj
