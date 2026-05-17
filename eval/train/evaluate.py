"""Train-slice evaluator for CaptainCook4D task graph generation.

This file is intentionally self-contained within eval/train/. It scores the
same prediction artifacts as scripts/evaluate.sh, but only on the public train
slice of CaptainCook4D scenarios.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np


SLICE_NAME = "train"
OUTPUT_PATH = "scoring/scores_train.json"
SCENARIOS = {
    # Public half of the 24 CaptainCook4D procedures. The split is balanced by
    # graph scale: 12 scenarios, 200 GT nodes, and 237 GT edges.
    "Broccolistirfry",
    "Capresebruschetta",
    "Cheesepimiento",
    "Dressedupmeatballs",
    "Herbomeletwithfriedtomatoes",
    "Microwaveeggsandwich",
    "Microwavefrenchtoast",
    "Microwavemugpizza",
    "Mugcake",
    "Pinwheels",
    "Sautedmushrooms",
    "Tomatomozzarellasalad",
}


@dataclass
class TaskGraphData:
    scenario_name: str
    num_steps: int
    sequences: List[List[int]]
    gt_adj_matrix: np.ndarray
    step_names: Dict[int, str]
    gt_node_count: int


def load_captaincook4d(repo_data_path: str) -> List[TaskGraphData]:
    annotations_path = os.path.join(repo_data_path, "task_graph_generation_annotations.json")
    gt_graphs_path = os.path.join(repo_data_path, "ground_truth_task_graphs")

    with open(annotations_path) as f:
        annotations = json.load(f)

    annotations_json = annotations["annotations"]
    taxonomy_json = annotations["taxonomy"]
    results: List[TaskGraphData] = []

    gt_files = sorted(filename for filename in os.listdir(gt_graphs_path) if filename.endswith(".json"))
    for gt_file in gt_files:
        activity_name = gt_file.replace(".json", "")
        with open(os.path.join(gt_graphs_path, gt_file)) as f:
            task_graph_json = json.load(f)

        graph = nx.DiGraph()
        for node_str, label in task_graph_json["steps"].items():
            graph.add_node(int(node_str), label=label)
        for dst, src in task_graph_json["edges"]:
            graph.add_edge(src, dst)

        n_gt_nodes = len(graph.nodes)
        n_matrix = n_gt_nodes + 2
        step_names = {int(node_str): label for node_str, label in task_graph_json["steps"].items()}

        taxonomy_scenario = taxonomy_json.get(activity_name, {})
        description_to_tax_id = {
            step_info["name"]: int(tax_id_str)
            for tax_id_str, step_info in taxonomy_scenario.items()
        }
        id_to_node = {}
        for node in graph.nodes:
            label = graph.nodes[node]["label"]
            if label in description_to_tax_id:
                id_to_node[description_to_tax_id[label]] = node

        train_sequences: List[List[int]] = []
        for video_data in annotations_json.values():
            if video_data["scenario"].lower().replace(" ", "") != activity_name:
                continue
            seq: List[int] = []
            for segment in video_data["segments"]:
                step_id = segment["step_id"]
                if step_id not in seq:
                    seq.append(step_id)
            if seq and seq not in train_sequences:
                train_sequences.append(seq)

        if not train_sequences:
            continue

        converted_seqs: List[List[int]] = []
        for seq in train_sequences:
            converted = [0, 1]
            for tax_id in seq:
                gt_node = id_to_node.get(tax_id)
                if gt_node is not None:
                    converted.append(gt_node + 1)
            converted.append(n_gt_nodes)
            converted.append(n_matrix - 1)
            converted_seqs.append(converted)

        gt_adj = np.zeros((n_gt_nodes, n_gt_nodes), dtype=np.float32)
        for dst, src in task_graph_json["edges"]:
            if dst < n_gt_nodes and src < n_gt_nodes:
                gt_adj[src, dst] = 1.0

        results.append(
            TaskGraphData(
                scenario_name=activity_name.replace("_", " ").title(),
                num_steps=n_gt_nodes - 2,
                sequences=converted_seqs,
                gt_adj_matrix=gt_adj,
                step_names=step_names,
                gt_node_count=n_gt_nodes,
            )
        )

    return results


def compute_precision_recall_f1(gt_adj: np.ndarray, pred_adj: np.ndarray) -> Tuple[float, float, float]:
    gt_flat = gt_adj.astype(np.int32).flatten()
    pred_flat = pred_adj.astype(np.int32).flatten()

    tp = int(np.sum((gt_flat == 1) & (pred_flat == 1)))
    fp = int(np.sum((gt_flat == 0) & (pred_flat == 1)))
    fn = int(np.sum((gt_flat == 1) & (pred_flat == 0)))

    precision = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def delete_redundant_edges(graph: nx.DiGraph) -> None:
    graph_copy = copy.deepcopy(graph)
    for current_node in graph.nodes:
        for successor in list(graph_copy.successors(current_node)):
            if graph.has_edge(current_node, successor):
                graph.remove_edge(current_node, successor)
                if not nx.has_path(graph, current_node, successor):
                    graph.add_edge(current_node, successor)


def add_boundary_edges(graph: nx.DiGraph, id_start: int, id_end: int) -> None:
    in_degree_zeros = [
        node
        for node in graph.nodes
        if graph.in_degree(node) == 0 and node != id_end and node != id_start
    ]
    for node in in_degree_zeros:
        graph.add_edge(id_end, node)

    out_degree_zeros = [
        node
        for node in graph.nodes
        if graph.out_degree(node) == 0 and node != id_end and node != id_start
    ]
    for node in out_degree_zeros:
        graph.add_edge(node, id_start)


def delete_cycles(graph: nx.DiGraph, scores: np.ndarray) -> None:
    while not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break
        cycle_edges = [(edge[0], edge[1]) for edge in cycle]
        edge_to_remove = min(
            cycle_edges,
            key=lambda edge: float(scores[edge[0], edge[1]]),
        )
        graph.remove_edge(*edge_to_remove)


def post_process_pred_adj(pred_core: np.ndarray, n_gt: int) -> np.ndarray:
    threshold = 1.0 / n_gt
    pred_binary = np.where(pred_core < threshold, 0, 1).astype(float)
    graph = nx.DiGraph(pred_binary)

    add_boundary_edges(graph, id_start=0, id_end=n_gt - 1)
    delete_redundant_edges(graph)
    delete_cycles(graph, pred_core)

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for src, dst in graph.edges():
        pred_adj[src, dst] = 1.0
    return pred_adj


def compute_topological_distance(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Compute a normalized partial-order distance between two DAGs.
    Uses disagreement in transitive reachability, which captures wrong,
    missing, or spurious precedence relations beyond direct edge overlap.
    Lower is better; 0 = identical topological structure.
    """
    try:
        gt_g = nx.DiGraph(gt_adj)
        pred_g = nx.DiGraph(pred_adj)

        if len(gt_g) <= 1:
            return 0.0

        if nx.is_directed_acyclic_graph(gt_g):
            gt_closure = nx.transitive_closure_dag(gt_g)
        else:
            gt_closure = nx.transitive_closure(gt_g)

        if nx.is_directed_acyclic_graph(pred_g):
            pred_closure = nx.transitive_closure_dag(pred_g)
        else:
            pred_closure = nx.transitive_closure(pred_g)

        nodes = sorted(set(gt_closure.nodes()) | set(pred_closure.nodes()))
        pair_count = 0
        disagreements = 0
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                pair_count += 1
                gt_has_order = gt_closure.has_edge(src, dst)
                pred_has_order = pred_closure.has_edge(src, dst)
                if gt_has_order != pred_has_order:
                    disagreements += 1

        return float(disagreements / pair_count) if pair_count else 0.0
    except Exception:
        return 0.5  # Default to moderate distance on failure


def compute_edge_density_ratio(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Compute a symmetric density-match ratio in [0, 1].
    1.0 = perfect density match; values decrease for under- or over-connection.
    """
    gt_edges = np.sum(gt_adj > 0)
    pred_edges = np.sum(pred_adj > 0)
    if gt_edges == 0:
        return 1.0 if pred_edges == 0 else 0.0
    if pred_edges == 0:
        return 0.0
    return min(gt_edges, pred_edges) / max(gt_edges, pred_edges)


def compute_path_coverage(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """
    Compute fraction of GT directed paths covered by predicted graph.
    A path is a sequence of edges forming a chain in topological order.
    Measures structural correctness beyond edge overlap.
    """
    try:
        gt_g = nx.DiGraph(gt_adj)
        pred_g = nx.DiGraph(pred_adj)

        # Find all source-to-sink paths in GT
        sources = [n for n in gt_g.nodes() if gt_g.in_degree(n) == 0]
        sinks = [n for n in gt_g.nodes() if gt_g.out_degree(n) == 0]

        if not sources or not sinks:
            return 0.5

        # Sample paths (full enumeration can be expensive)
        all_paths = []
        for src in sources:
            for sink in sinks:
                try:
                    for path in nx.all_simple_paths(gt_g, src, sink):
                        all_paths.append(tuple(sorted(zip(path[:-1], path[1:]))))
                except nx.NetworkXNoPath:
                    continue
                except nx.NetworkXError:
                    continue

        if not all_paths:
            return 0.5

        # Count how many GT paths are also present in prediction
        covered = 0
        pred_edges = set(pred_g.edges())
        for path_edges in all_paths:
            if all(edge in pred_edges for edge in path_edges):
                covered += 1

        return covered / len(all_paths)
    except Exception:
        return 0.5


def evaluate_from_adjacency(pred_core: np.ndarray, gt_adj: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate a predicted adjacency matrix against ground truth.
    Returns (precision, recall, f1, std_f1, edge_density_ratio, path_coverage, topological_distance).
    """
    pred_adj = post_process_pred_adj(pred_core, gt_adj.shape[0])
    precision, recall, f1 = compute_precision_recall_f1(gt_adj, pred_adj)

    # Structural metrics
    density = compute_edge_density_ratio(gt_adj, pred_adj)
    path_cov = compute_path_coverage(gt_adj, pred_adj)
    topo_dist = compute_topological_distance(gt_adj, pred_adj)

    return precision, recall, f1, density, path_cov, topo_dist


def strip_matrix_tokens(sequences: List[List[int]], n_gt: int) -> List[List[int]]:
    gt_sequences: List[List[int]] = []
    for seq in sequences:
        gt_seq = [node - 1 for node in seq if 1 <= node < n_gt]
        if gt_seq:
            gt_sequences.append(gt_seq)
    return gt_sequences


def build_transition_matrix(sequences: List[List[int]], n_gt: int) -> np.ndarray:
    transitions = np.zeros((n_gt, n_gt), dtype=np.float64)
    for seq in sequences:
        for index in range(len(seq) - 1):
            transitions[seq[index], seq[index + 1]] += 1.0

    row_sums = transitions.sum(axis=1)
    for index in range(n_gt):
        if row_sums[index] > 0:
            transitions[index, :] /= row_sums[index]
    return transitions


def sorted_preconditions(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
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


def sorted_futures(transitions: np.ndarray, step_id: int) -> Dict[int, float]:
    row = transitions[step_id, :]
    scores = {action: float(row[action]) for action in range(len(row)) if row[action] > 0}
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def remove_cycles_with_min_prob(graph: nx.DiGraph, transitions: np.ndarray) -> None:
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
    del threshold
    if not sequences:
        return np.zeros((2, 2), dtype=np.float32)
    if n_gt is None:
        raise ValueError("count_based_with_threshold requires n_gt")

    gt_sequences = strip_matrix_tokens(sequences, n_gt)
    transitions = build_transition_matrix(gt_sequences, n_gt)

    graph = nx.DiGraph()
    for action in range(n_gt):
        graph.add_node(action)

    for action in range(n_gt):
        preconditions = sorted_preconditions(transitions, action)
        if not preconditions:
            continue
        max_score = max(preconditions.values())
        for precondition, score in preconditions.items():
            if score == max_score and precondition != action:
                graph.add_edge(precondition, action)

    remove_cycles_with_min_prob(graph, transitions)

    for node in list(graph.nodes()):
        if graph.in_degree(node) == 0 and node != 0:
            for precondition in sorted_preconditions(transitions, node):
                graph.add_edge(precondition, node)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(precondition, node)
                else:
                    break

    for node in list(graph.nodes()):
        if graph.out_degree(node) == 0 and node != (n_gt - 1):
            for future in sorted_futures(transitions, node):
                graph.add_edge(node, future)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(node, future)
                else:
                    break

    in_degree_zeros = [
        node for node in graph.nodes if graph.in_degree(node) == 0 and node not in (0, n_gt - 1)
    ]
    for node in in_degree_zeros:
        graph.add_edge(0, node)

    out_degree_zeros = [
        node for node in graph.nodes if graph.out_degree(node) == 0 and node not in (0, n_gt - 1)
    ]
    for node in out_degree_zeros:
        graph.add_edge(node, n_gt - 1)

    graph = graph.reverse(copy=True)
    delete_redundant_edges(graph)

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for src, dst in graph.edges():
        pred_adj[src, dst] = 1.0
    return pred_adj


def safe_prediction_name(scenario_name: str) -> str:
    return scenario_name.lower().replace(" ", "_")


def evaluate_count_based(
    sequences: List[List[int]],
    gt_adj: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float, float, float]:
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
    do_predictions_dir: Optional[str] = None,
    method: str = "do",
    threshold: float = 0.5,
    allow_partial: bool = False,
    min_do_seeds: Optional[int] = None,
) -> Dict[str, Any]:
    results: Dict[str, Dict[str, float]] = {}
    seed_f1_totals: Dict[int, List[float]] = {}

    for scenario_data in data:
        scenario_name = scenario_data.scenario_name
        if method == "count_based":
            precision, recall, f1, density, path_cov, topo_dist = evaluate_count_based(
                scenario_data.sequences,
                scenario_data.gt_adj_matrix,
                threshold,
            )
        elif method == "do":
            if not do_predictions_dir or not os.path.exists(do_predictions_dir):
                raise FileNotFoundError(
                    f"DO evaluation requires a valid predictions directory, got: {do_predictions_dir}"
                )
            prediction_file = os.path.join(
                do_predictions_dir,
                f"{safe_prediction_name(scenario_name)}_predictions.npz",
            )
            if not os.path.exists(prediction_file):
                if allow_partial:
                    continue
                raise FileNotFoundError(f"Missing DO prediction file: {prediction_file}")

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
                    f"{prediction_file} has {len(pred_cores)} seeds, but {min_do_seeds} are required."
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
        else:
            raise ValueError(f"Unknown method: {method}")

        results[scenario_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "edge_density_ratio": float(density),
            "path_coverage": float(path_cov),
            "topological_distance": float(topo_dist),
        }

    if not results:
        raise ValueError(f"No scenarios were evaluated for {SLICE_NAME} slice")

    all_precision = [result["precision"] for result in results.values()]
    all_recall = [result["recall"] for result in results.values()]
    all_f1 = [result["f1"] for result in results.values()]
    all_density = [result["edge_density_ratio"] for result in results.values()]
    all_path_cov = [result["path_coverage"] for result in results.values()]
    all_topo_dist = [result["topological_distance"] for result in results.values()]
    sorted_f1 = sorted(all_f1)
    worst_quartile_count = max(1, int(np.ceil(len(sorted_f1) * 0.25)))
    worst_quartile_f1 = float(np.mean(sorted_f1[:worst_quartile_count]))
    if method == "do" and len(seed_f1_totals) > 1:
        seed_means = [float(np.mean(seed_scores)) for _, seed_scores in sorted(seed_f1_totals.items())]
        std_f1 = float(np.std(seed_means))
    else:
        std_f1 = 0.0
    return {
        "per_scenario": results,
        "aggregate": {
            "mean_precision": float(np.mean(all_precision)),
            "mean_recall": float(np.mean(all_recall)),
            "mean_f1": float(np.mean(all_f1)),
            "std_precision": float(np.std(all_precision)),
            "std_recall": float(np.std(all_recall)),
            "std_f1": std_f1,
            "worst_quartile_f1": worst_quartile_f1,
            "edge_density_ratio": float(np.mean(all_density)),
            "path_coverage": float(np.mean(all_path_cov)),
            "topological_distance": float(np.mean(all_topo_dist)),
        },
    }


def write_scores_json(
    results: Dict[str, Any],
    experiment_name: str,
    output_path: str = OUTPUT_PATH,
    dataset_override: Optional[str] = None,
    method_override: Optional[str] = None,
    artifact_path: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = dataset_override or (
        "captaincook4d_task_graph" if "captaincook" in experiment_name.lower() else experiment_name
    )
    method_name = method_override or (
        "count_based" if "count" in experiment_name.lower() else experiment_name
    )

    scores: Dict[str, Any] = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            scores = json.load(f)

    scores["slice"] = SLICE_NAME
    scores["slice_manifest"] = {
        "slice": SLICE_NAME,
        "dataset": dataset,
        "split_rule": "balanced fixed 12/12 split over CaptainCook4D procedures by GT node and edge counts",
        "scenario_count": len(results["per_scenario"]),
        "scenarios": sorted(results["per_scenario"].keys()),
    }
    scores.setdefault("experiments", {})
    scores["experiments"].setdefault(dataset, {"results": {}})
    scores.setdefault("method_artifacts", {})
    scores["method_artifacts"][method_name] = {
        "artifact_type": "built_in_count_baseline"
        if method_name == "count_based"
        else "npz_prediction_directory",
        "checkpoint_dir": artifact_path or "",
    }

    aggregate = results["aggregate"]
    scores["experiments"][dataset]["results"][method_name] = {
        "precision": aggregate["mean_precision"],
        "recall": aggregate["mean_recall"],
        "f1": aggregate["mean_f1"],
        "std_f1": aggregate["std_f1"],
        "worst_quartile_f1": aggregate["worst_quartile_f1"],
        "edge_density_ratio": aggregate["edge_density_ratio"],
        "path_coverage": aggregate["path_coverage"],
        "topological_distance": aggregate["topological_distance"],
    }

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")


def select_slice(data: Iterable[TaskGraphData]) -> List[TaskGraphData]:
    selected = [scenario for scenario in data if scenario.scenario_name in SCENARIOS]
    missing = sorted(SCENARIOS - {scenario.scenario_name for scenario in selected})
    if missing:
        raise ValueError(f"Missing {SLICE_NAME} scenarios from loaded data: {missing}")
    return selected


def main(argv: List[str]) -> None:
    method_name = argv[1] if len(argv) > 1 else "captaincook4d_do"
    checkpoint_dir = argv[2] if len(argv) > 2 else "checkpoints/"
    min_seeds = int(argv[3]) if len(argv) > 3 else 5

    repo_path = "/home/user/data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d"
    data = select_slice(load_captaincook4d(repo_path))
    print(f"Loaded {len(data)} {SLICE_NAME} scenarios")

    if method_name == "captaincook4d_partial_compare":
        do_results = evaluate_all_scenarios(
            data,
            do_predictions_dir=checkpoint_dir,
            method="do",
            allow_partial=True,
            min_do_seeds=min_seeds,
        )
        completed = set(do_results["per_scenario"].keys())
        subset_data = [scenario for scenario in data if scenario.scenario_name in completed]
        count_results = evaluate_all_scenarios(subset_data, method="count_based")
        print(f"Completed {SLICE_NAME} subset: {len(completed)} scenarios -> {sorted(completed)}")
        write_scores_json(
            count_results,
            "captaincook4d_count_partial",
            dataset_override="captaincook4d_task_graph",
            method_override="count_based",
        )
        write_scores_json(
            do_results,
            "captaincook4d_do_partial",
            dataset_override="captaincook4d_task_graph",
            method_override="do",
            artifact_path=checkpoint_dir,
        )
    elif "count" in method_name:
        results = evaluate_all_scenarios(data, method="count_based")
        print(f"Count-Based {SLICE_NAME}: F1={results['aggregate']['mean_f1']:.1f}")
        write_scores_json(
            results,
            method_name,
            dataset_override="captaincook4d_task_graph",
            method_override="count_based",
        )
    else:
        if checkpoint_dir in ("", "checkpoints/"):
            raise SystemExit("DO evaluation requires an explicit checkpoint/results directory.")
        results = evaluate_all_scenarios(data, do_predictions_dir=checkpoint_dir, method="do")
        print(f"DO {SLICE_NAME}: F1={results['aggregate']['mean_f1']:.1f}")
        write_scores_json(
            results,
            method_name,
            dataset_override="captaincook4d_task_graph",
            method_override=method_name,
            artifact_path=checkpoint_dir,
        )

    print(f"Scores written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main(sys.argv)
