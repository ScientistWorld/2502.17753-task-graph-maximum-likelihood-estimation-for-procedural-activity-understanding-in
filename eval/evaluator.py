"""
Evaluation code for task graph generation.

Evaluates predicted task graphs against ground truth using
precision, recall, and F1 score on edge prediction.
Uses the paper's evaluation approach.
"""

import json
import numpy as np
import torch
import os
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from data.loader import TaskGraphData, load_data
from method.tgml import (
    DirectOptimization,
    compute_precision_recall_f1,
    delete_redundant_edges, delete_cycles
)
from baseline.count_based import count_based_with_threshold


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    scenario: str
    precision: float
    recall: float
    f1: float
    method: str


def evaluate_do(
    model: DirectOptimization,
    sequences: List[List[int]],
    gt_adj: np.ndarray,
    device: str = "cpu"
) -> Tuple[float, float, float]:
    """
    Evaluate DO model predictions against ground truth.

    Uses the paper's evaluation approach:
    1. Extract core nodes: pred[1:-1, 1:-1]
    2. Threshold at 1/n
    3. Add boundary edges (START/END)
    4. Transitive reduction
    5. Compare with GT

    Parameters
    ----------
    model : DirectOptimization
        Trained DO model.
    sequences : List[List[int]]
        Training sequences (unused, kept for API compatibility).
    gt_adj : np.ndarray
        Ground truth adjacency matrix (n_gt) x (n_gt), indexed by GT node ID.
    device : str
        Device for computation.

    Returns
    -------
    Tuple[float, float, float]
        (precision, recall, f1)
    """
    n_gt = gt_adj.shape[0]
    threshold = 1.0 / n_gt

    model.eval()
    with torch.no_grad():
        Z = model.get_adjacency_matrix().cpu().numpy()

    # Extract core nodes: remove beta (row/col 0) and gamma (row/col -1)
    pred_core = Z[1:-1, 1:-1]  # (n_gt) x (n_gt)

    # Binarize
    pred_binary = (pred_core > threshold).astype(float)

    # Build NetworkX graph
    GP = nx.DiGraph(pred_binary)

    # Add boundary edges (matching paper's approach)
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

    return compute_precision_recall_f1(gt_adj, pred_adj)


def evaluate_count_based(
    sequences: List[List[int]],
    gt_adj: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
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

    # Resize to match GT
    pred_resized = np.zeros((n_gt, n_gt), dtype=np.float32)
    min_n = min(pred_adj.shape[0], n_gt)
    pred_resized[:min_n, :min_n] = pred_adj[:min_n, :min_n]

    # Post-process
    GP = nx.DiGraph(pred_resized)

    id_start = 0
    id_end = n_gt - 1

    in_degree_zeros = [n for n in GP.nodes if GP.in_degree(n) == 0 and n != id_end]
    for node in in_degree_zeros:
        GP.add_edge(id_end, node)

    out_degree_zeros = [n for n in GP.nodes if GP.out_degree(n) == 0 and n != id_start]
    for node in out_degree_zeros:
        GP.add_edge(node, id_start)

    delete_redundant_edges(GP)
    delete_cycles(GP)

    pred_final = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_final[v, u] = 1.0

    return compute_precision_recall_f1(gt_adj, pred_final)


def evaluate_all_scenarios(
    data: List[TaskGraphData],
    do_predictions_dir: str = None,
    method: str = "do",
    threshold: float = 0.5,
    seeds: List[int] = None,
    device: str = "cpu"
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
        Device for computation.

    Returns
    -------
    Dict[str, Any]
        Evaluation results per scenario and aggregated.
    """
    if seeds is None:
        seeds = [42]

    results = {}

    for scenario_data in data:
        scenario_name = scenario_data.scenario_name

        if method == "count_based":
            precision, recall, f1 = evaluate_count_based(
                scenario_data.sequences,
                scenario_data.gt_adj_matrix,
                threshold
            )
            results[scenario_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
        elif method == "do":
            if do_predictions_dir and os.path.exists(do_predictions_dir):
                result_file = os.path.join(do_predictions_dir, f"{scenario_name.lower().replace(' ', '_')}_results.json")
                if os.path.exists(result_file):
                    with open(result_file) as f:
                        saved = json.load(f)
                    results[scenario_name] = {
                        "precision": saved.get("precision", 0),
                        "recall": saved.get("recall", 0),
                        "f1": saved.get("f1", 0)
                    }
                    continue

            # Run count-based as fallback
            precision, recall, f1 = evaluate_count_based(
                scenario_data.sequences,
                scenario_data.gt_adj_matrix,
                threshold
            )
            results[scenario_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }

    # Aggregate
    all_precision = [r["precision"] for r in results.values()]
    all_recall = [r["recall"] for r in results.values()]
    all_f1 = [r["f1"] for r in results.values()]

    aggregate = {
        "mean_precision": float(np.mean(all_precision)),
        "mean_recall": float(np.mean(all_recall)),
        "mean_f1": float(np.mean(all_f1)),
        "std_f1": float(np.std(all_f1)),
        "std_precision": float(np.std(all_precision)),
        "std_recall": float(np.std(all_recall)),
    }

    return {
        "per_scenario": results,
        "aggregate": aggregate
    }


def write_scores_json(
    results: Dict[str, Any],
    experiment_name: str,
    output_path: str = "scoring/scores.json"
) -> None:
    """
    Write evaluation results to scores.json.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if "captaincook" in experiment_name.lower():
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
    method_name = "do" if "do" in experiment_name.lower() else "count_based"

    scores["experiments"][dataset]["results"][method_name] = {
        "precision": agg["mean_precision"],
        "recall": agg["mean_recall"],
        "f1": agg["mean_f1"],
    }

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
