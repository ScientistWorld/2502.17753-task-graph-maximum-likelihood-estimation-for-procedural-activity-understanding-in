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
    delete_redundant_edges,
    delete_cycles,
    add_boundary_edges,
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


def evaluate_from_adjacency(
    pred_core: np.ndarray,
    gt_adj: np.ndarray
) -> Tuple[float, float, float]:
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
    Tuple[float, float, float]
        (precision, recall, f1)
    """
    n_gt = gt_adj.shape[0]
    pred_adj = post_process_pred_adj(pred_core, n_gt)
    return compute_precision_recall_f1(gt_adj, pred_adj)


def post_process_pred_adj(
    pred_core: np.ndarray,
    n_gt: int
) -> np.ndarray:
    """
    Post-process predicted adjacency matrix following the paper's approach:
    1. Binarize at 1/n threshold
    2. Add boundary edges
    3. Transitive reduction
    4. Cycle removal

    Parameters
    ----------
    pred_core : np.ndarray
        Raw predicted scores (n_gt x n_gt).
    n_gt : int
        Number of GT nodes.

    Returns
    -------
    np.ndarray
        Processed binary adjacency matrix.
    """
    threshold = 1.0 / n_gt
    pred_binary = (pred_core > threshold).astype(float)

    GP = nx.DiGraph(pred_binary)

    id_start = 0
    id_end = n_gt - 1

    add_boundary_edges(GP, id_start=id_start, id_end=id_end)

    delete_redundant_edges(GP)
    delete_cycles(GP)

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_adj[v, u] = 1.0

    return pred_adj


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

    add_boundary_edges(GP, id_start=id_start, id_end=id_end)

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
        Device for computation (unused, for API compatibility).

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
            if not do_predictions_dir or not os.path.exists(do_predictions_dir):
                raise FileNotFoundError(
                    f"DO evaluation requires a valid predictions directory, got: {do_predictions_dir}"
                )
            prediction_file = os.path.join(
                do_predictions_dir,
                f"{scenario_name.lower().replace(' ', '_')}_predictions.npz"
            )
            if not os.path.exists(prediction_file):
                raise FileNotFoundError(
                    f"Missing DO prediction file for {scenario_name}: {prediction_file}"
                )
            saved = np.load(prediction_file)
            pred_cores = saved["pred_cores"] if "pred_cores" in saved else saved["pred_core"][None, ...]
            per_seed_metrics = [
                evaluate_from_adjacency(pred_core, scenario_data.gt_adj_matrix)
                for pred_core in pred_cores
            ]
            precision = float(np.mean([metric[0] for metric in per_seed_metrics]))
            recall = float(np.mean([metric[1] for metric in per_seed_metrics]))
            f1 = float(np.mean([metric[2] for metric in per_seed_metrics]))
            results[scenario_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
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
    if method_name == "do":
        scores["experiments"][dataset]["results"][method_name]["std_f1"] = agg["std_f1"]

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
