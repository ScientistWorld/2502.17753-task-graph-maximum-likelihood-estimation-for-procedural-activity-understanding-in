"""
Count-Based baseline for task graph learning.

This baseline estimates precondition relationships by counting co-occurrences
of steps in action sequences. If step A frequently appears before step B
across multiple sequences, A is likely a precondition of B.

Based on the paper's approach.
"""

import numpy as np
from typing import List, Dict, Tuple
import networkx as nx


def count_based_baseline(
    sequences: List[List[int]],
    threshold: float = 0.0
) -> np.ndarray:
    """
    Compute adjacency matrix using the count-based method.

    For each pair of nodes (i, j), count how many times j appears before i
    across all sequences. The weight is the count normalized by frequency.

    Parameters
    ----------
    sequences : List[List[int]]
        List of sequences. Each sequence is a list of step IDs (1..n for steps,
        0=START, n+1=END). Note: step IDs are 1-indexed in sequences.
    threshold : float
        Minimum count threshold (fraction of total sequences) for an edge.

    Returns
    -------
    np.ndarray
        Adjacency matrix (n+2) x (n+2).
    """
    if not sequences:
        return np.zeros((2, 2))

    # Determine matrix size
    all_ids = set()
    for seq in sequences:
        all_ids.update(seq)
    max_id = max(all_ids)
    n_matrix = max_id + 1  # 0-indexed

    # Count co-occurrences
    # cooccur[i, j] = number of times node j appears before node i
    cooccur = np.zeros((n_matrix, n_matrix), dtype=np.float32)

    for seq in sequences:
        # Remove beta and gamma (0 and max_id)
        # Actually, let's keep all
        seq = list(seq)
        for i in range(len(seq)):
            for j in range(i):
                # seq[i] appears after seq[j]
                cooccur[seq[i], seq[j]] += 1

    # The paper's count-based method: edge from j to i exists if
    # j frequently appears before i in the sequences
    # We normalize by the total number of occurrences of i
    adj = np.zeros((n_matrix, n_matrix), dtype=np.float32)

    for i in range(n_matrix):
        total_before_i = cooccur[i, :].sum()
        if total_before_i > 0:
            # Edge weight = count / total appearances
            adj[i, :] = cooccur[i, :] / total_before_i

    return adj


def count_based_with_threshold(
    sequences: List[List[int]],
    threshold: float = 0.5,
    n_gt: int = None
) -> np.ndarray:
    """
    Count-based baseline with explicit thresholding.

    An edge j->i exists if at least `threshold` fraction of sequences
    have j appearing before i.

    Parameters
    ----------
    sequences : List[List[int]]
        Training sequences using matrix indices [0=beta, 1..n=GTnodes, n+1=gamma].
    threshold : float
        Fraction threshold for edge existence.
    n_gt : int
        Number of GT nodes (for correct output size). If None, inferred.
    """
    if not sequences:
        return np.zeros((2, 2))

    # Determine GT size from sequences
    # Matrix indices: 0=beta, 1..n=GT, n+1=gamma
    # GT node IDs are matrix_index - 1, ranging from 0 to (n_gt - 1)
    max_id = max(max(seq) for seq in sequences)

    if n_gt is not None:
        n_matrix = n_gt + 1  # matrix has one more for beta
    else:
        # Infer: max_id = n_gt (gamma) → n_gt = max_id - 1
        # But could also be n_matrix - 1 if max_id is a step
        n_matrix = max_id + 1

    # Count occurrences
    cooccur = np.zeros((n_matrix, n_matrix), dtype=np.float32)
    total_occurrences = np.zeros(n_matrix, dtype=np.float32)

    for seq in sequences:
        for i in range(len(seq)):
            if seq[i] >= n_matrix:
                continue
            total_occurrences[seq[i]] += 1
            for j in range(i):
                if seq[j] < n_matrix:
                    cooccur[seq[i], seq[j]] += 1

    # Binary adjacency (matrix indices)
    adj = np.zeros((n_matrix, n_matrix), dtype=np.float32)
    for i in range(n_matrix):
        if total_occurrences[i] > 0:
            for j in range(n_matrix):
                fraction = cooccur[i, j] / total_occurrences[i]
                if fraction >= threshold:
                    adj[i, j] = 1.0

    # Extract core (GT nodes only, 1..n-1) → convert to GT node IDs (0..n-2)
    if n_gt is not None:
        # Extract [1:n_gt+1, 1:n_gt+1] and shift to GT node IDs
        adj_gt = adj[1:n_gt+1, 1:n_gt+1]
        return adj_gt

    return adj


def count_based_from_paper(
    sequences: List[List[int]]
) -> Dict[int, List[int]]:
    """
    Count-based baseline as described in the paper (Ashutosh et al., 2024).
    Matches video segments and counts co-occurrences.

    Returns a dictionary: node_id -> list of precondition IDs.
    """
    if not sequences:
        return {}

    # Build eligibility vectors
    eligibility = {}  # node -> list of binary vectors
    nodes = sorted(set(node for seq in sequences for node in seq))

    for seq in sequences:
        for i, node in enumerate(seq):
            if node not in eligibility:
                eligibility[node] = []
            ev = np.zeros(len(nodes))
            for j, n in enumerate(nodes):
                if n in seq[:i]:
                    ev[j] = 1
            eligibility[node].append(ev)

    # Count how often each potential precondition appears with each node
    preconditions = {}
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    for node, evs in eligibility.items():
        evs_array = np.array(evs)
        # For each potential precondition
        node_idx = node_to_idx.get(node, -1)
        if node_idx < 0:
            continue

        node_preconditions = []
        for j, other_node in enumerate(nodes):
            if other_node == node:
                continue
            # Count: how often does other_node appear before node?
            count = evs_array[:, j].sum()
            # Also consider: how often does node appear without other_node?
            # The paper uses a simple approach: if j appears before node in
            # most sequences where node appears, it's a precondition
            if count >= len(evs) * 0.5:  # threshold of 50%
                node_preconditions.append(other_node)

        preconditions[node] = node_preconditions

    return preconditions


def adjacency_to_networkx_count_based(
    adj: np.ndarray,
    threshold: float = 0.5
) -> nx.DiGraph:
    """
    Convert count-based adjacency matrix to NetworkX graph.
    """
    binary = (adj >= threshold).astype(float)
    G = nx.DiGraph(binary)

    # Remove self-loops
    for i in range(min(G.number_of_nodes(), adj.shape[0])):
        if G.has_edge(i, i):
            G.remove_edge(i, i)

    return G
