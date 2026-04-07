"""Count-Based baseline matching the paper's transition-graph procedure."""

from __future__ import annotations

import copy
from typing import Dict, List

import networkx as nx
import numpy as np


def _strip_matrix_tokens(sequences: List[List[int]], n_gt: int) -> List[List[int]]:
    """Convert matrix-index sequences [beta, START, ..., END, gamma] to GT node ids."""
    gt_sequences: List[List[int]] = []
    for seq in sequences:
        gt_seq = [node - 1 for node in seq[1:-1] if 1 <= node <= n_gt]
        if gt_seq:
            gt_sequences.append(gt_seq)
    return gt_sequences


def _build_transition_matrix(sequences: List[List[int]], n_gt: int) -> np.ndarray:
    """Reproduce the transition matrix used by the released Count-Based baseline."""
    transitions = np.zeros((n_gt, n_gt), dtype=np.float32)
    for seq in sequences:
        deduped: List[int] = []
        for step in seq:
            if step not in deduped:
                deduped.append(step)
        for i in range(len(deduped) - 1):
            transitions[deduped[i], deduped[i + 1]] += 1.0

    row_sums = transitions.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        transitions = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums > 0)
    return transitions


def _sorted_preconditions(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
    """Return P(pred=A | current=step_id), sorted descending."""
    denom = transitions[:, step_id].sum()
    if denom <= 0:
        return {}
    scores = {
        action: float(transitions[action, step_id] / denom)
        for action in range(transitions.shape[0])
        if transitions[action, step_id] > 0
    }
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _sorted_futures(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
    """Return P(next=A | current=step_id), sorted descending."""
    scores = {
        action: float(transitions[step_id, action])
        for action in range(transitions.shape[0])
        if transitions[step_id, action] > 0
    }
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _remove_redundant_edges(graph: nx.DiGraph) -> None:
    graph_copy = copy.deepcopy(graph)
    for current_node in graph.nodes:
        for successor in list(graph_copy.successors(current_node)):
            graph.remove_edge(current_node, successor)
            if not nx.has_path(graph, current_node, successor):
                graph.add_edge(current_node, successor)


def _remove_cycles_with_min_transition(graph: nx.DiGraph, transitions: np.ndarray) -> None:
    for cycle in list(nx.simple_cycles(graph)):
        if len(cycle) < 2:
            if graph.has_edge(cycle[0], cycle[0]):
                graph.remove_edge(cycle[0], cycle[0])
            continue

        min_edge = None
        min_prob = float("inf")
        extended = cycle + [cycle[0]]
        for src, dst in zip(extended[:-1], extended[1:]):
            prob = float(transitions[src, dst])
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
    Reproduce the paper's Count-Based baseline from action sequences.

    The `threshold` argument is accepted for API compatibility but the released
    baseline keeps the highest-probability predecessor(s) per node rather than
    applying a global threshold.
    """
    del threshold

    if not sequences:
        return np.zeros((2, 2), dtype=np.float32)

    if n_gt is None:
        raise ValueError("count_based_with_threshold requires n_gt")

    gt_sequences = _strip_matrix_tokens(sequences, n_gt)
    transitions = _build_transition_matrix(gt_sequences, n_gt)

    # Build the graph in the same direction as the released baseline before its final reverse:
    # predecessor -> current step.
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_gt))

    for action in range(n_gt):
        preconditions = _sorted_preconditions(transitions, action)
        if not preconditions:
            continue
        max_score = max(preconditions.values())
        for precondition, score in preconditions.items():
            if score == max_score and precondition != action:
                graph.add_edge(precondition, action)

    _remove_cycles_with_min_transition(graph, transitions)

    # Match the released fallback logic for isolated nodes while preserving acyclicity.
    for node in list(graph.nodes()):
        if graph.in_degree(node) == 0 and node != 0:
            for precondition in _sorted_preconditions(transitions, node):
                graph.add_edge(precondition, node)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(precondition, node)
                else:
                    break

    for node in list(graph.nodes()):
        if graph.out_degree(node) == 0 and node != (n_gt - 1):
            for future in _sorted_futures(transitions, node):
                graph.add_edge(node, future)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(node, future)
                else:
                    break

    # Boundary edges are inserted before reversing, matching the released code.
    in_degree_zeros = [node for node in graph.nodes if graph.in_degree(node) == 0 and node not in (0, n_gt - 1)]
    for node in in_degree_zeros:
        graph.add_edge(0, node)

    out_degree_zeros = [node for node in graph.nodes if graph.out_degree(node) == 0 and node not in (0, n_gt - 1)]
    for node in out_degree_zeros:
        graph.add_edge(node, n_gt - 1)

    graph = graph.reverse(copy=True)
    _remove_redundant_edges(graph)

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for src, dst in graph.edges():
        pred_adj[dst, src] = 1.0
    return pred_adj
