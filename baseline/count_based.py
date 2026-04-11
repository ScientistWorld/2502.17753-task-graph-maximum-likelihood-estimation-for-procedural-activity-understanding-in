"""
Count-Based baseline reproducing the paper's transition-graph procedure.
Based on the official code in task-graph-learning/baselines/ and _transition_graph.py.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import networkx as nx
import numpy as np


def _strip_matrix_tokens(sequences: List[List[int]], n_gt: int) -> List[List[int]]:
    """
    Convert matrix-index sequences [beta, START, ..., END, gamma] to GT node ids.

    Matrix indices: 0=beta, 1=START, 2..(n_gt-1)=steps, n_gt=END, n_gt+1=gamma
    GT node IDs: 0=START, 1..(n_gt-2)=steps, n_gt-1=END
    Matrix index m maps to GT node ID m-1 for m in [1, n_gt-1].
    """
    gt_sequences: List[List[int]] = []
    for seq in sequences:
        # Exclude: 0 (beta), n_gt (END), n_gt+1 (gamma); include: 1..n_gt-1
        gt_seq = [node - 1 for node in seq if 1 <= node < n_gt]
        if gt_seq:
            gt_sequences.append(gt_seq)
    return gt_sequences


def _build_transition_matrix(sequences: List[List[int]], n_gt: int) -> np.ndarray:
    """
    Build transition matrix matching the official TransitionGraph approach.
    Transitions are consecutive step pairs in the sequences.
    """
    transitions = np.zeros((n_gt, n_gt), dtype=np.float64)
    for seq in sequences:
        for i in range(len(seq) - 1):
            transitions[seq[i], seq[i + 1]] += 1.0
    # Row-normalize: P(succ=j | pred=i)
    row_sums = transitions.sum(axis=1)
    for i in range(n_gt):
        if row_sums[i] > 0:
            transitions[i, :] /= row_sums[i]
    return transitions


def _sorted_preconditions(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
    """
    Get preconditions of step_id sorted by P(pred=action | succ=step_id) descending.
    Matches official: conditional_prob_A_given_X = transition_matrix.at[action_A, step_id] / col_sum
    """
    col = transitions[:, step_id]
    col_sum = col.sum()
    if col_sum <= 0:
        return {}
    scores = {action: float(col[action] / col_sum) for action in range(len(col)) if col[action] > 0}
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _sorted_futures(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
    """
    Get future steps of step_id sorted by P(succ=action | pred=step_id) descending.
    Matches official: conditional_prob_A_given_X = transition_matrix.at[step_id, action_A]
    """
    row = transitions[step_id, :]
    scores = {action: float(row[action]) for action in range(len(row)) if row[action] > 0}
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _delete_redundant_edges(graph: nx.DiGraph) -> None:
    """Transitive reduction - matches official delete_redundant_edges."""
    G_copy = copy.deepcopy(graph)
    for current_node in graph.nodes:
        for successor in list(G_copy.successors(current_node)):
            graph.remove_edge(current_node, successor)
            if not nx.has_path(graph, current_node, successor):
                graph.add_edge(current_node, successor)


def _remove_cycles_with_min_prob(graph: nx.DiGraph, transitions: np.ndarray) -> None:
    """Remove cycles by removing the edge with minimum transition probability."""
    for cycle in list(nx.simple_cycles(graph)):
        if len(cycle) < 2:
            if graph.has_edge(cycle[0], cycle[0]):
                graph.remove_edge(cycle[0], cycle[0])
            continue
        # Edge with minimum transition probability in the cycle
        min_prob = float('inf')
        min_edge = None
        for i in range(len(cycle)):
            src, dst = cycle[i], cycle[(i + 1) % len(cycle)]
            prob = transitions[src, dst]
            if prob < min_prob:
                min_prob = prob
                min_edge = (src, dst)
        if min_edge is not None and graph.has_edge(*min_edge):
            graph.remove_edge(*min_edge)


def count_based_with_threshold(
    sequences: List[List[int]],
    threshold: float = 0.5,
    n_gt: int | None = None,
) -> np.ndarray:
    """
    Reproduce the official Count-Based baseline from action sequences.

    The `threshold` argument is accepted for API compatibility but the baseline
    keeps the highest-probability predecessor(s) per node rather than applying
    a global threshold.
    """
    del threshold

    if not sequences:
        return np.zeros((2, 2), dtype=np.float32)

    if n_gt is None:
        raise ValueError("count_based_with_threshold requires n_gt")

    gt_sequences = _strip_matrix_tokens(sequences, n_gt)
    transitions = _build_transition_matrix(gt_sequences, n_gt)

    # Build graph (precondition -> current step), matching official create_networkx_graph
    graph = nx.DiGraph()
    for action in range(n_gt):
        graph.add_node(action)

    for action in range(n_gt):
        preconditions = _sorted_preconditions(transitions, action)
        if not preconditions:
            continue
        max_score = max(preconditions.values())
        for precondition, score in preconditions.items():
            if score == max_score and precondition != action:
                graph.add_edge(precondition, action)

    # Remove cycles by minimum transition probability
    _remove_cycles_with_min_prob(graph, transitions)

    # Handle isolated nodes (preconditions): if node has no incoming edges
    for node in list(graph.nodes()):
        if graph.in_degree(node) == 0 and node != 0:
            preconditions = _sorted_preconditions(transitions, node)
            for precondition in preconditions:
                graph.add_edge(precondition, node)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(precondition, node)
                else:
                    break

    # Handle isolated nodes (futures): if node has no outgoing edges
    for node in list(graph.nodes()):
        if graph.out_degree(node) == 0 and node != (n_gt - 1):
            futures = _sorted_futures(transitions, node)
            for future in futures:
                graph.add_edge(node, future)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(node, future)
                else:
                    break

    # Add boundary edges (before reversing), matching official
    in_degree_zeros = [node for node in graph.nodes if graph.in_degree(node) == 0 and node not in (0, n_gt - 1)]
    for node in in_degree_zeros:
        graph.add_edge(0, node)

    out_degree_zeros = [node for node in graph.nodes if graph.out_degree(node) == 0 and node not in (0, n_gt - 1)]
    for node in out_degree_zeros:
        graph.add_edge(node, n_gt - 1)

    # Reverse all edges (matching official: G.reverse())
    graph = graph.reverse(copy=True)

    # Remove redundant edges (transitive reduction)
    _delete_redundant_edges(graph)

    # Build adjacency matrix in graph orientation (adj[u,v]=1 means edge u->v)
    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for src, dst in graph.edges():
        pred_adj[src, dst] = 1.0

    return pred_adj