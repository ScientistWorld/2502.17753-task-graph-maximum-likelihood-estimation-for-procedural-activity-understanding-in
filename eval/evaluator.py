"""
Evaluation code for task graph generation.

Evaluates predicted task graphs against ground truth using
precision, recall, and F1 score on edge prediction.
Uses the paper's evaluation approach.

This module has NO imports from method/, ensuring separation
between problem definition (eval/) and solution (method/).
"""

import json
import numpy as np
import os
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from data.loader import TaskGraphData, load_data
from eval.metrics import (
    compute_precision_recall_f1,
    compute_edge_density_ratio,
    compute_path_coverage,
    compute_topological_distance,
    delete_redundant_edges,
    delete_cycles,
    add_boundary_edges,
)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    scenario: str
    precision: float
    recall: float
    f1: float
    method: str


def evaluate_from_adjacency(
    pred_core: np.ndarray,
    gt_adj: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate a predicted adjacency matrix against ground truth.

    Parameters
    ----------
    pred_core : np.ndarray
        Core predicted adjacency matrix (n_gt x n_gt), raw scores.
    gt_adj : np.ndarray
        Ground truth adjacency matrix (n_gt x n_gt).

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        (precision, recall, f1, edge_density_ratio, path_coverage, topological_distance)
    """
    n_gt = gt_adj.shape[0]
    pred_adj = post_process_pred_adj(pred_core, n_gt)
    precision, recall, f1 = compute_precision_recall_f1(gt_adj, pred_adj)
    density = compute_edge_density_ratio(gt_adj, pred_adj)
    path_cov = compute_path_coverage(gt_adj, pred_adj)
    topo_dist = compute_topological_distance(gt_adj, pred_adj)
    return precision, recall, f1, density, path_cov, topo_dist


def post_process_pred_adj(
    pred_core: np.ndarray,
    n_gt: int
) -> np.ndarray:
    """
    Post-process predicted adjacency matrix following the paper's approach:
    1. Binarize at 1/n_gt threshold, matching the official 1/(num_nodes - 2)
    2. Add boundary edges
    3. Transitive reduction
    4. Cycle removal

    Parameters
    ----------
    pred_core : np.ndarray
        Raw predicted scores (n_gt x n_gt).
    n_gt : int
        Number of GT nodes (including START and END).

    Returns
    -------
    np.ndarray
        Processed binary adjacency matrix.
    """
    # Official threshold: 1/(num_nodes - 2), where num_nodes = n_gt + 2.
    threshold = 1.0 / n_gt
    pred_binary = np.where(pred_core < threshold, 0, 1).astype(float)

    # Build graph matching official: pred_binary[row, col] = 1 means edge col -> row
    # DiGraph(matrix) creates graph with edge col -> row where matrix[row, col] > 0
    GP = nx.DiGraph(pred_binary)

    id_start = 0
    id_end = n_gt - 1

    add_boundary_edges(GP, id_start=id_start, id_end=id_end)
    delete_redundant_edges(GP)
    delete_cycles(GP, pred_core)

    # pred_adj has graph orientation: adj[u,v]=1 means edge u->v (u is precondition of v)
    # graph edge u->v means adj[u,v]=1
    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_adj[u, v] = 1.0

    return pred_adj


def _strip_matrix_tokens(sequences: List[List[int]], n_gt: int) -> List[List[int]]:
    gt_sequences: List[List[int]] = []
    for seq in sequences:
        gt_seq = [node - 1 for node in seq if 1 <= node < n_gt]
        if gt_seq:
            gt_sequences.append(gt_seq)
    return gt_sequences


def _build_transition_matrix(sequences: List[List[int]], n_gt: int) -> np.ndarray:
    transitions = np.zeros((n_gt, n_gt), dtype=np.float64)
    for seq in sequences:
        for index in range(len(seq) - 1):
            transitions[seq[index], seq[index + 1]] += 1.0
    row_sums = transitions.sum(axis=1)
    for index in range(n_gt):
        if row_sums[index] > 0:
            transitions[index, :] /= row_sums[index]
    return transitions


def _sorted_preconditions(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
    column = transitions[:, step_id]
    column_sum = column.sum()
    if column_sum <= 0:
        return {}
    scores = {
        action: float(column[action] / column_sum)
        for action in range(len(column))
        if column[action] > 0
    }
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _sorted_futures(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
    row = transitions[step_id, :]
    scores = {action: float(row[action]) for action in range(len(row)) if row[action] > 0}
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _remove_cycles_with_min_prob(graph: nx.DiGraph, transitions: np.ndarray) -> None:
    for cycle in list(nx.simple_cycles(graph)):
        if len(cycle) < 2:
            if graph.has_edge(cycle[0], cycle[0]):
                graph.remove_edge(cycle[0], cycle[0])
            continue
        min_prob = float("inf")
        min_edge = None
        for index in range(len(cycle)):
            src, dst = cycle[index], cycle[(index + 1) % len(cycle)]
            probability = transitions[src, dst]
            if probability < min_prob:
                min_prob = probability
                min_edge = (src, dst)
        if min_edge is not None and graph.has_edge(*min_edge):
            graph.remove_edge(*min_edge)


def count_based_with_threshold(
    sequences: List[List[int]],
    threshold: float = 0.5,
    n_gt: Optional[int] = None,
) -> np.ndarray:
    """Count-Based baseline scorer, kept in eval/ so evaluation is self-contained."""
    del threshold
    if not sequences:
        return np.zeros((2, 2), dtype=np.float32)
    if n_gt is None:
        raise ValueError("count_based_with_threshold requires n_gt")

    gt_sequences = _strip_matrix_tokens(sequences, n_gt)
    transitions = _build_transition_matrix(gt_sequences, n_gt)

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

    _remove_cycles_with_min_prob(graph, transitions)

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

    in_degree_zeros = [
        node for node in graph.nodes
        if graph.in_degree(node) == 0 and node not in (0, n_gt - 1)
    ]
    for node in in_degree_zeros:
        graph.add_edge(0, node)

    out_degree_zeros = [
        node for node in graph.nodes
        if graph.out_degree(node) == 0 and node not in (0, n_gt - 1)
    ]
    for node in out_degree_zeros:
        graph.add_edge(node, n_gt - 1)

    graph = graph.reverse(copy=True)
    delete_redundant_edges(graph)

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for src, dst in graph.edges():
        pred_adj[src, dst] = 1.0
    return pred_adj


def evaluate_count_based(
    sequences: List[List[int]],
    gt_adj: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate count-based baseline predictions.

    Parameters
    ----------
    sequences : List[List[int]]
        Training sequences (matrix indices).
    gt_adj : np.ndarray
        Ground truth adjacency matrix (n_gt) x (n_gt).
    threshold : float
        Threshold for count-based method.

    Returns
    -------
    Tuple[float, float, float]
        (precision, recall, f1)
    """
    n_gt = gt_adj.shape[0]
    pred_adj = count_based_with_threshold(sequences, threshold=threshold, n_gt=n_gt)

    if pred_adj.shape != gt_adj.shape:
        pred_resized = np.zeros((n_gt, n_gt), dtype=np.float32)
        min_n = min(pred_adj.shape[0], n_gt)
        pred_resized[:min_n, :min_n] = pred_adj[:min_n, :min_n]
        pred_adj = pred_resized

    precision, recall, f1 = compute_precision_recall_f1(gt_adj, pred_adj)
    density = compute_edge_density_ratio(gt_adj, pred_adj)
    path_cov = compute_path_coverage(gt_adj, pred_adj)
    topo_dist = compute_topological_distance(gt_adj, pred_adj)
    return precision, recall, f1, density, path_cov, topo_dist


def evaluate_all_scenarios(
    data: List[TaskGraphData],
    do_predictions_dir: str = None,
    method: str = "do",
    threshold: float = 0.5,
    seeds: List[int] = None,
    device: str = "cpu",
    allow_partial: bool = False,
    min_do_seeds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate all scenarios.

    Parameters
    ----------
    data : List[TaskGraphData]
        List of scenario data.
    do_predictions_dir : str
        Directory containing DO model predictions.
    method : str
        Method to evaluate: "do" or "count_based".
    threshold : float
        Threshold for count-based (ignored for DO).
    seeds : List[int]
        List of seeds for DO (for averaging).
    device : str
        Device for computation (unused, for API compatibility).

    Returns
    -------
    Dict[str, Any]
        Evaluation results per scenario and aggregated.
    """
    if seeds is None:
        seeds = [42]

    results = {}
    seed_f1_totals: Dict[int, List[float]] = {}

    for scenario_data in data:
        scenario_name = scenario_data.scenario_name

        if method == "count_based":
            precision, recall, f1, density, path_cov, topo_dist = evaluate_count_based(
                scenario_data.sequences,
                scenario_data.gt_adj_matrix,
                threshold
            )
            results[scenario_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "edge_density_ratio": float(density),
                "path_coverage": float(path_cov),
                "topological_distance": float(topo_dist),
            }
        elif method == "do":
            if not do_predictions_dir or not os.path.exists(do_predictions_dir):
                raise FileNotFoundError(
                    f"DO evaluation requires a valid predictions directory, got: {do_predictions_dir}"
                )
            prediction_file = os.path.join(
                do_predictions_dir,
                f"{scenario_name.lower().replace(' ', '_')}_predictions.npz"
            )
            if not os.path.exists(prediction_file):
                if allow_partial:
                    continue
                raise FileNotFoundError(
                    f"Missing DO prediction file for {scenario_name}: {prediction_file}"
                )
            saved = np.load(prediction_file)
            pred_cores = saved["pred_cores"] if "pred_cores" in saved else saved["pred_core"][None, ...]
            seed_ids = (
                [int(seed) for seed in saved["seeds"].tolist()]
                if "seeds" in saved
                else list(range(len(pred_cores)))
            )
            if min_do_seeds is not None and len(pred_cores) < min_do_seeds:
                if allow_partial:
                    continue
                raise ValueError(
                    f"Prediction file for {scenario_name} has only {len(pred_cores)} seeds, "
                    f"but {min_do_seeds} are required."
                )
            per_seed_metrics = [
                evaluate_from_adjacency(pred_core, scenario_data.gt_adj_matrix)
                for pred_core in pred_cores
            ]
            for seed_id, metric in zip(seed_ids, per_seed_metrics):
                seed_f1_totals.setdefault(seed_id, []).append(float(metric[2]))
            precision = float(np.mean([metric[0] for metric in per_seed_metrics]))
            recall = float(np.mean([metric[1] for metric in per_seed_metrics]))
            f1 = float(np.mean([metric[2] for metric in per_seed_metrics]))
            density = float(np.mean([metric[3] for metric in per_seed_metrics]))
            path_cov = float(np.mean([metric[4] for metric in per_seed_metrics]))
            topo_dist = float(np.mean([metric[5] for metric in per_seed_metrics]))
            results[scenario_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "edge_density_ratio": density,
                "path_coverage": path_cov,
                "topological_distance": topo_dist,
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    # Aggregate
    if not results:
        raise ValueError(f"No scenarios were evaluated for method={method}")

    all_precision = [r["precision"] for r in results.values()]
    all_recall = [r["recall"] for r in results.values()]
    all_f1 = [r["f1"] for r in results.values()]
    all_density = [r["edge_density_ratio"] for r in results.values()]
    all_path_cov = [r["path_coverage"] for r in results.values()]
    all_topo_dist = [r["topological_distance"] for r in results.values()]
    sorted_f1 = sorted(all_f1)
    quartile_count = max(1, len(sorted_f1) // 4)
    worst_quartile_f1 = float(np.mean(sorted_f1[:quartile_count]))

    if method == "do" and len(seed_f1_totals) > 1:
        seed_means = [
            float(np.mean(seed_scores))
            for _, seed_scores in sorted(seed_f1_totals.items())
        ]
        std_f1 = float(np.std(seed_means))
    else:
        std_f1 = 0.0

    aggregate = {
        "mean_precision": float(np.mean(all_precision)),
        "mean_recall": float(np.mean(all_recall)),
        "mean_f1": float(np.mean(all_f1)),
        "std_f1": std_f1,
        "std_precision": float(np.std(all_precision)),
        "std_recall": float(np.std(all_recall)),
        "worst_quartile_f1": worst_quartile_f1,
        "edge_density_ratio": float(np.mean(all_density)),
        "path_coverage": float(np.mean(all_path_cov)),
        "topological_distance": float(np.mean(all_topo_dist)),
    }

    return {
        "per_scenario": results,
        "aggregate": aggregate
    }


def write_scores_json(
    results: Dict[str, Any],
    experiment_name: str,
    output_path: str = "scoring/scores.json",
    dataset_override: Optional[str] = None,
    method_override: Optional[str] = None,
) -> None:
    """
    Write evaluation results to scores.json.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if dataset_override is not None:
        dataset = dataset_override
    elif "captaincook" in experiment_name.lower():
        dataset = "captaincook4d_task_graph"
    elif "egoper" in experiment_name.lower():
        dataset = "egoper_task_graph"
    elif "egoprocel" in experiment_name.lower():
        dataset = "egoprocel_task_graph"
    else:
        dataset = experiment_name

    scores = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            scores = json.load(f)

    if "experiments" not in scores:
        scores["experiments"] = {}

    if dataset not in scores["experiments"]:
        scores["experiments"][dataset] = {"results": {}}

    agg = results["aggregate"]
    if method_override is not None:
        method_name = method_override
    else:
        experiment_key = experiment_name.lower()
        if "count" in experiment_key:
            method_name = "count_based"
        elif experiment_key in {"do", "captaincook4d_do"} or experiment_key.endswith("_do"):
            method_name = "do"
        else:
            for prefix in ("captaincook4d_", "egoper_", "egoprocel_"):
                if experiment_key.startswith(prefix):
                    experiment_key = experiment_key[len(prefix):]
                    break
            method_name = experiment_key

    scores["experiments"][dataset]["results"][method_name] = {
        "precision": agg["mean_precision"],
        "recall": agg["mean_recall"],
        "f1": agg["mean_f1"],
        "std_f1": agg["std_f1"],
        "worst_quartile_f1": agg["worst_quartile_f1"],
        "edge_density_ratio": agg["edge_density_ratio"],
        "path_coverage": agg["path_coverage"],
        "topological_distance": agg["topological_distance"],
    }

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
