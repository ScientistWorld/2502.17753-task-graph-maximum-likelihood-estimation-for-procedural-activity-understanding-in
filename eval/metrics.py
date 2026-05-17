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


def compute_edge_density_ratio(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Compute a symmetric edge-density match ratio in [0, 1].

    This is a constraint metric: it discourages methods from improving recall by
    predicting much denser graphs than the ground truth.
    """
    gt_edges = np.sum(gt_adj > 0)
    pred_edges = np.sum(pred_adj > 0)
    if gt_edges == 0:
        return 1.0 if pred_edges == 0 else 0.0
    if pred_edges == 0:
        return 0.0
    return float(min(gt_edges, pred_edges) / max(gt_edges, pred_edges))


def compute_path_coverage(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Fraction of ground-truth source-to-sink paths exactly covered by prediction.

    Edge F1 is the paper's primary metric. This structural metric is an
    additional gym constraint so future methods preserve useful DAG structure.
    """
    try:
        gt_g = nx.DiGraph(gt_adj)
        pred_g = nx.DiGraph(pred_adj)
        sources = [node for node in gt_g.nodes() if gt_g.in_degree(node) == 0]
        sinks = [node for node in gt_g.nodes() if gt_g.out_degree(node) == 0]
        if not sources or not sinks:
            return 0.5

        all_paths = []
        for src in sources:
            for sink in sinks:
                try:
                    for path in nx.all_simple_paths(gt_g, src, sink):
                        all_paths.append(tuple(zip(path[:-1], path[1:])))
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue

        if not all_paths:
            return 0.5

        pred_edges = set(pred_g.edges())
        covered = sum(
            1 for path_edges in all_paths
            if all(edge in pred_edges for edge in path_edges)
        )
        return float(covered / len(all_paths))
    except Exception:
        return 0.5


def compute_topological_distance(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Normalized disagreement between predicted and GT transitive reachability.

    Lower is better. This captures wrong partial orders beyond direct edge
    overlap while remaining independent of any method implementation.
    """
    try:
        gt_g = nx.DiGraph(gt_adj)
        pred_g = nx.DiGraph(pred_adj)
        if len(gt_g) <= 1:
            return 0.0

        gt_closure = (
            nx.transitive_closure_dag(gt_g)
            if nx.is_directed_acyclic_graph(gt_g)
            else nx.transitive_closure(gt_g)
        )
        pred_closure = (
            nx.transitive_closure_dag(pred_g)
            if nx.is_directed_acyclic_graph(pred_g)
            else nx.transitive_closure(pred_g)
        )

        nodes = sorted(set(gt_closure.nodes()) | set(pred_closure.nodes()))
        pair_count = 0
        disagreements = 0
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                pair_count += 1
                if gt_closure.has_edge(src, dst) != pred_closure.has_edge(src, dst):
                    disagreements += 1
        return float(disagreements / pair_count) if pair_count else 0.0
    except Exception:
        return 0.5


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


def delete_cycles(G: nx.DiGraph, scores: np.ndarray | None = None) -> None:
    """Remove cycles by deleting the lowest-score edge in each detected cycle."""
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation="original")
            cycle_edges = [(edge[0], edge[1]) for edge in cycle]
            if scores is None:
                edge_to_remove = cycle_edges[-1]
            else:
                edge_to_remove = min(
                    cycle_edges,
                    key=lambda edge: float(scores[edge[0], edge[1]]),
                )
            G.remove_edge(*edge_to_remove)
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
    # Official threshold: 1/(num_nodes - 2), where num_nodes = n_gt + 2.
    threshold = 1.0 / n_gt

    # Binarize matching official: np.where(pred_core < threshold, 0, 1)
    pred_binary = np.where(pred_core < threshold, 0, 1).astype(float)

    # Build graph matching official: pred_binary[row, col] = 1 means edge col -> row
    GP = nx.DiGraph(pred_binary)

    # Add boundary edges
    id_start = 0  # GT node ID
    id_end = n_gt - 1  # GT node ID

    add_boundary_edges(GP, id_start=id_start, id_end=id_end)

    delete_redundant_edges(GP)
    delete_cycles(GP, pred_core)

    # pred_adj has graph orientation: adj[u,v]=1 means edge u->v (u is precondition of v)
    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_adj[u, v] = 1.0

    return pred_adj
