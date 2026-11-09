"""Public (train/dev) evaluator for task-graph generation. Two experiments:

  captaincook4d_task_graph         12 public CaptainCook4D procedures
  noisy_demonstrations_task_graph  40 public procedures whose demonstrations are
                                   perturbed with the paper's insert/delete/replace
                                   noise model (data/noisy_demos/train.json)

Self-contained in eval/train/ (it shares no code with the hidden test). Ground
truth and demonstrations of the public procedures are read from data/ (CaptainCook4D
via scripts/download.sh; the noisy benchmark from data/noisy_demos/). The hidden
test applies exactly the same contract and metrics to held-out procedures.

Every directory under the predictions root (default: checkpoints/) that holds at
least one ``*_predictions.npz`` file is scored as one method, keyed by the
directory name. A method must provide, for each procedure of an experiment,
``<procedure>_predictions.npz`` with an array ``pred_cores`` of shape
(5, n, n): five independent trials of a weighted adjacency matrix with finite
values in [0, 1], indexed by the procedure's node ids (entry [i, j] scores
"node j is a precondition of node i"). Any missing procedure, wrong shape,
non-finite or out-of-range value marks the method as failed on that experiment;
a failed method receives the floor value of every scored metric.

Each trial is binarised at the fixed threshold 1/n and post-processed exactly as
in the paper (boundary START/END completion, transitive reduction, removal of
the lowest-scored edge of every cycle), then compared with the ground truth.

Usage (no arguments scores every method on disk):
    python3 eval/train/evaluate.py [--predictions-root DIR] [--output FILE]
                                   [--details-output FILE]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "data" / "downloads" / "Differentiable-Task-Graph-Learning" / "data" / "captaincook4d"
NOISY_FILE = WORKSPACE / "data" / "noisy_demos" / "train.json"
DEFAULT_PREDICTIONS_ROOT = WORKSPACE / "checkpoints"
DEFAULT_OUTPUT = WORKSPACE / "scoring" / "scores_train.json"

SLICE_NAME = "train"
SCENARIO_KEYS = (
    # Public half of the 24 CaptainCook4D procedures (balanced against the
    # held-out half by ground-truth node and edge counts).
    "broccolistirfry",
    "capresebruschetta",
    "cheesepimiento",
    "dressedupmeatballs",
    "herbomeletwithfriedtomatoes",
    "microwaveeggsandwich",
    "microwavefrenchtoast",
    "microwavemugpizza",
    "mugcake",
    "pinwheels",
    "sautedmushrooms",
    "tomatomozzarellasalad",
)
N_TRIALS = 5
BASELINE_KEY = "count_based"
PREDICTION_SUFFIX = "_predictions.npz"
METRIC_NAMES = (
    "precision",
    "recall",
    "f1",
    "worst_quartile_f1",
    "edge_density_ratio",
    "path_coverage",
    "topological_distance",
)
# Floor of every scored metric: what a failed run receives.
FAILED_SCORES = {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "worst_quartile_f1": 0.0,
    "edge_density_ratio": 0.0,
    "path_coverage": 0.0,
    "topological_distance": 1.0,
}


class InvalidSubmission(Exception):
    """Raised when a method's artifacts violate the prediction contract."""


@dataclass
class Procedure:
    key: str
    name: str
    n: int
    gt_adj: np.ndarray
    sequences: List[List[int]]


def load_captaincook4d_procedures() -> List[Procedure]:
    with open(DATA_DIR / "task_graph_generation_annotations.json") as f:
        annotations = json.load(f)
    videos = annotations["annotations"]
    taxonomy = annotations["taxonomy"]
    procedures: List[Procedure] = []
    for key in SCENARIO_KEYS:
        with open(DATA_DIR / "ground_truth_task_graphs" / f"{key}.json") as f:
            graph_json = json.load(f)
        labels = {int(node): label for node, label in graph_json["steps"].items()}
        n = len(labels)
        description_to_tax_id = {info["name"]: int(tax_id) for tax_id, info in taxonomy.get(key, {}).items()}
        tax_id_to_node = {}
        for node in sorted(labels):
            if labels[node] in description_to_tax_id:
                tax_id_to_node[description_to_tax_id[labels[node]]] = node
        raw_sequences: List[List[int]] = []
        for video in videos.values():
            if video["scenario"].lower().replace(" ", "") != key:
                continue
            seq: List[int] = []
            for segment in video["segments"]:
                if segment["step_id"] not in seq:
                    seq.append(segment["step_id"])
            if seq and seq not in raw_sequences:
                raw_sequences.append(seq)
        # Matrix indices: 0=beta, node+1 for node ids (1=START, n=END), n+1=gamma.
        sequences = [[0, 1] + [tax_id_to_node[t] + 1 for t in seq if t in tax_id_to_node] + [n, n + 1]
                     for seq in raw_sequences]
        gt_adj = np.zeros((n, n), dtype=np.float32)
        for a, b in graph_json["edges"]:
            gt_adj[int(b), int(a)] = 1.0
        procedures.append(Procedure(key=key, name=key.replace("_", " ").title(), n=n,
                                    gt_adj=gt_adj, sequences=sequences))
    return procedures


def load_noisy_procedures() -> List[Procedure]:
    with open(NOISY_FILE) as f:
        blob = json.load(f)
    procedures: List[Procedure] = []
    for key in sorted(blob["procedures"]):
        entry = blob["procedures"][key]
        n = int(entry["n_nodes"])
        gt_adj = np.zeros((n, n), dtype=np.float32)
        for a, b in entry["edges"]:
            gt_adj[int(b), int(a)] = 1.0
        sequences = [[0, 1] + [int(v) + 1 for v in demo] + [n, n + 1] for demo in entry["demonstrations"]]
        procedures.append(Procedure(key=key, name=key, n=n, gt_adj=gt_adj, sequences=sequences))
    return procedures


EXPERIMENT_LOADERS = {
    "captaincook4d_task_graph": load_captaincook4d_procedures,
    "noisy_demonstrations_task_graph": load_noisy_procedures,
}


# --------------------------------------------------------------------------
# Graph post-processing (paper Section III-E) and metrics
# --------------------------------------------------------------------------

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
        node for node in graph.nodes
        if graph.in_degree(node) == 0 and node != id_end and node != id_start
    ]
    for node in in_degree_zeros:
        graph.add_edge(id_end, node)
    out_degree_zeros = [
        node for node in graph.nodes
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
        edge_to_remove = min(cycle_edges, key=lambda edge: float(scores[edge[0], edge[1]]))
        graph.remove_edge(*edge_to_remove)


def post_process_pred_adj(pred_core: np.ndarray, n: int) -> np.ndarray:
    threshold = 1.0 / n
    pred_binary = np.where(pred_core < threshold, 0, 1).astype(float)
    graph = nx.DiGraph(pred_binary)
    add_boundary_edges(graph, id_start=0, id_end=n - 1)
    delete_redundant_edges(graph)
    delete_cycles(graph, pred_core)
    pred_adj = np.zeros((n, n), dtype=np.float32)
    for src, dst in graph.edges():
        pred_adj[src, dst] = 1.0
    return pred_adj


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


def compute_edge_density_ratio(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """min(|E_pred|, |E_gt|) / max(|E_pred|, |E_gt|); 1.0 = same number of edges."""
    gt_edges = int(np.sum(gt_adj > 0))
    pred_edges = int(np.sum(pred_adj > 0))
    if gt_edges == 0:
        return 1.0 if pred_edges == 0 else 0.0
    if pred_edges == 0:
        return 0.0
    return min(gt_edges, pred_edges) / max(gt_edges, pred_edges)


def compute_path_coverage(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """Fraction of ground-truth source-to-sink paths whose every edge is predicted."""
    gt_g = nx.DiGraph(gt_adj)
    pred_edges = set(nx.DiGraph(pred_adj).edges())
    sources = [node for node in gt_g.nodes() if gt_g.in_degree(node) == 0]
    sinks = [node for node in gt_g.nodes() if gt_g.out_degree(node) == 0]
    total = 0
    covered = 0
    for src in sources:
        for sink in sinks:
            if src == sink:
                continue
            for path in nx.all_simple_paths(gt_g, src, sink):
                total += 1
                if all(edge in pred_edges for edge in zip(path[:-1], path[1:])):
                    covered += 1
    return covered / total if total else 0.0


def compute_topological_distance(gt_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    """Fraction of ordered node pairs whose reachability differs (0 = same partial order)."""
    gt_closure = nx.transitive_closure(nx.DiGraph(gt_adj), reflexive=False)
    pred_closure = nx.transitive_closure(nx.DiGraph(pred_adj), reflexive=False)
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


def graph_metrics(gt_adj: np.ndarray, pred_adj: np.ndarray) -> Dict[str, float]:
    precision, recall, f1 = compute_precision_recall_f1(gt_adj, pred_adj)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edge_density_ratio": compute_edge_density_ratio(gt_adj, pred_adj),
        "path_coverage": compute_path_coverage(gt_adj, pred_adj),
        "topological_distance": compute_topological_distance(gt_adj, pred_adj),
    }


# --------------------------------------------------------------------------
# Built-in Count-Based baseline (fixed by the evaluator)
# --------------------------------------------------------------------------

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
    scores = {action: float(column[action] / column_sum)
              for action in range(len(column)) if column[action] > 0}
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


def count_based_graph(sequences: List[List[int]], n_gt: int) -> np.ndarray:
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

    for node in [n for n in graph.nodes if graph.in_degree(n) == 0 and n not in (0, n_gt - 1)]:
        graph.add_edge(0, node)
    for node in [n for n in graph.nodes if graph.out_degree(n) == 0 and n not in (0, n_gt - 1)]:
        graph.add_edge(node, n_gt - 1)

    graph = graph.reverse(copy=True)
    delete_redundant_edges(graph)
    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for src, dst in graph.edges():
        pred_adj[src, dst] = 1.0
    return pred_adj


# --------------------------------------------------------------------------
# Method discovery, validation and aggregation
# --------------------------------------------------------------------------

def discover_methods(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir())
            if d.is_dir() and any(d.glob("*" + PREDICTION_SUFFIX))]


def load_trials(path: Path, n: int) -> np.ndarray:
    if not path.is_file():
        raise InvalidSubmission(f"missing {path.name}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "pred_cores" not in archive.files:
                raise InvalidSubmission(f"{path.name}: no 'pred_cores' array")
            cores = np.array(archive["pred_cores"])
    except InvalidSubmission:
        raise
    except Exception as exc:  # unreadable / pickled / corrupt archive
        raise InvalidSubmission(f"{path.name}: unreadable ({type(exc).__name__}: {exc})")
    if cores.dtype.kind not in "biuf":
        raise InvalidSubmission(f"{path.name}: non-numeric dtype {cores.dtype}")
    if cores.shape != (N_TRIALS, n, n):
        raise InvalidSubmission(
            f"{path.name}: pred_cores shape {cores.shape}, required {(N_TRIALS, n, n)}")
    cores = cores.astype(np.float64)
    if not np.all(np.isfinite(cores)):
        raise InvalidSubmission(f"{path.name}: non-finite values")
    if cores.min() < 0.0 or cores.max() > 1.0:
        raise InvalidSubmission(
            f"{path.name}: values outside [0, 1] (min {cores.min():.4g}, max {cores.max():.4g})")
    return cores


def aggregate(per_procedure: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(per_procedure)
    f1_sorted = sorted(per_procedure[k]["f1"] for k in keys)
    worst_count = max(1, int(math.ceil(len(f1_sorted) * 0.25)))
    out = {}
    for metric in ("precision", "recall", "f1", "edge_density_ratio", "path_coverage",
                   "topological_distance"):
        out[metric] = float(np.mean([per_procedure[k][metric] for k in keys]))
    out["worst_quartile_f1"] = float(np.mean(f1_sorted[:worst_count]))
    return {metric: out[metric] for metric in METRIC_NAMES}


def check_scores(scores: Dict[str, float]) -> None:
    for metric in METRIC_NAMES:
        value = scores[metric]
        if not math.isfinite(value):
            raise InvalidSubmission(f"non-finite {metric}")
        upper = 100.0 if metric in ("precision", "recall", "f1", "worst_quartile_f1") else 1.0
        if value < 0.0 or value > upper:
            raise InvalidSubmission(f"impossible {metric}={value}")


def score_method(method_dir: Path, procedures: List[Procedure]) -> Dict:
    per_procedure: Dict[str, Dict[str, float]] = {}
    trial_f1 = np.zeros((N_TRIALS, len(procedures)))
    for j, proc in enumerate(procedures):
        cores = load_trials(method_dir / f"{proc.key}{PREDICTION_SUFFIX}", proc.n)
        trial_metrics = [graph_metrics(proc.gt_adj, post_process_pred_adj(cores[t], proc.n))
                         for t in range(N_TRIALS)]
        per_procedure[proc.name] = {
            metric: float(np.mean([tm[metric] for tm in trial_metrics]))
            for metric in trial_metrics[0]
        }
        trial_f1[:, j] = [tm["f1"] for tm in trial_metrics]
    scores = aggregate(per_procedure)
    check_scores(scores)
    trial_means = trial_f1.mean(axis=1)
    return {
        "scores": scores,
        "per_procedure": per_procedure,
        "diagnostics": {
            "trial_mean_f1": [float(x) for x in trial_means],
            "trial_std_f1": float(np.std(trial_means)),
        },
    }


def score_baseline(procedures: List[Procedure]) -> Dict:
    per_procedure = {proc.name: graph_metrics(proc.gt_adj, count_based_graph(proc.sequences, proc.n))
                     for proc in procedures}
    return {"scores": aggregate(per_procedure), "per_procedure": per_procedure,
            "diagnostics": {"deterministic": True}}


def method_key(method_dir: Path) -> str:
    name = method_dir.name
    return name if name != BASELINE_KEY else f"{name}__submitted"


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details-output", type=Path, default=None)
    args = parser.parse_args(argv)

    method_dirs = discover_methods(args.predictions_root)
    experiments: Dict[str, Dict] = {}
    status: Dict[str, Dict] = {}
    diagnostics: Dict[str, Dict] = {}
    details: Dict[str, Dict] = {}
    manifest: Dict[str, Dict] = {}

    for experiment, loader in EXPERIMENT_LOADERS.items():
        procedures = loader()
        print(f"[{experiment}] {len(procedures)} public {SLICE_NAME} procedures")
        results: Dict[str, Dict[str, float]] = {}
        exp_status: Dict[str, Dict] = {}
        exp_diag: Dict[str, Dict] = {}
        exp_details: Dict[str, Dict] = {}
        for method_dir in method_dirs:
            key = method_key(method_dir)
            try:
                scored = score_method(method_dir, procedures)
                results[key] = scored["scores"]
                exp_status[key] = {"status": "ok", "artifact_dir": relative(method_dir)}
                exp_diag[key] = scored["diagnostics"]
                exp_details[key] = scored["per_procedure"]
                print(f"  {key}: F1={scored['scores']['f1']:.2f}")
            except Exception as exc:  # contract violation (InvalidSubmission) or unscorable artifact
                reason = str(exc) if isinstance(exc, InvalidSubmission) else f"unscorable: {type(exc).__name__}: {exc}"
                results[key] = dict(FAILED_SCORES)
                exp_status[key] = {"status": "failed", "reason": reason, "artifact_dir": relative(method_dir)}
                print(f"  {key}: FAILED ({reason})")

        baseline = score_baseline(procedures)
        results[BASELINE_KEY] = baseline["scores"]
        exp_status[BASELINE_KEY] = {"status": "ok", "artifact_dir": "built-in (eval/train)"}
        exp_diag[BASELINE_KEY] = baseline["diagnostics"]
        exp_details[BASELINE_KEY] = baseline["per_procedure"]
        print(f"  {BASELINE_KEY}: F1={baseline['scores']['f1']:.2f}")

        experiments[experiment] = {"results": results}
        status[experiment] = exp_status
        diagnostics[experiment] = exp_diag
        details[experiment] = exp_details
        manifest[experiment] = {"scenario_count": len(procedures),
                                "scenarios": sorted(proc.name for proc in procedures)}

    output = {
        "slice": SLICE_NAME,
        "slice_manifest": {
            "slice": SLICE_NAME,
            "split_rule": "fixed public/held-out split; CaptainCook4D balanced 12/12 by GT node and edge counts",
            "trials_per_procedure": N_TRIALS,
            "predictions_root": relative(args.predictions_root),
            "experiments": manifest,
        },
        "experiments": experiments,
        "method_status": status,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.details_output, "w") as f:
            json.dump({"per_procedure": details, "method_status": status}, f, indent=2)
            f.write("\n")
    print(f"Scores written to {relative(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
