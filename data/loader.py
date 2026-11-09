"""
Data loader for the TGML paper's datasets.

Supports:
- CaptainCook4D: loaded from the cloned repo's data directory
- Synthetic data: generated procedurally for testing

Key indexing convention (matching paper's train_with_gt.py):
- GT node IDs: sequential integers 0..(n-1) where 0=START, n-1=END, 1..(n-2)=steps
- Matrix indices: beta=0, GT0(START)=1, GT1=2, ..., GT(n-1)(END)=n, gamma=n+1
- Matrix size = len(GT.nodes) + 2 = n + 2
- Taxonomy IDs: arbitrary integers from the dataset's taxonomy (e.g., Coffee uses 56-71)
- Taxonomy → GT mapping via description string matching
- Edges [dst, src] in JSON mean src is a precondition of dst (src → dst in graph direction)
- In adjacency matrix A[row, col] = 1 means edge from col to row (col is precondition of row)
"""

import json
import numpy as np
import networkx as nx
import os
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class TaskGraphData:
    """Container for task graph data."""
    scenario_name: str
    num_steps: int  # number of actual key steps (excluding START/END), i.e., n-2
    sequences: List[List[int]]  # sequences using MATRIX indices [0=beta, 1..n=GTnodes, n+1=gamma]
    gt_adj_matrix: np.ndarray  # ground truth adjacency matrix (n) x (n), indexed by GT node ID
    step_names: Dict[int, str]  # GT_node_id -> step name
    gt_node_count: int  # total GT nodes = n (includes START and END)
    has_ground_truth: bool = True  # False for held-out procedures (edges withheld; zeros in gt_adj_matrix)


def load_captaincook4d(repo_data_path: str) -> List[TaskGraphData]:
    """
    Load CaptainCook4D dataset from the cloned repo.
    Returns list of TaskGraphData, one per scenario.

    Matrix indices: [0=beta, 1..n=GTnodes, n+1=gamma]
    where n = len(GT.nodes) = number of actual GT nodes (including START and END).
    """
    annotations_path = os.path.join(repo_data_path, "task_graph_generation_annotations.json")
    gt_graphs_path = os.path.join(repo_data_path, "ground_truth_task_graphs")

    with open(annotations_path) as f:
        annotations = json.load(f)

    annotations_json = annotations["annotations"]
    taxonomy_json = annotations["taxonomy"]

    results = []

    gt_files = sorted([f for f in os.listdir(gt_graphs_path) if f.endswith(".json")])

    for gt_file in gt_files:
        activity_name = gt_file.replace(".json", "")  # e.g., "coffee"

        # Load ground truth graph
        with open(os.path.join(gt_graphs_path, gt_file)) as f:
            task_graph_json = json.load(f)

        # Build NetworkX graph from GT
        GT = nx.DiGraph()
        for node_str, label in task_graph_json["steps"].items():
            GT.add_node(int(node_str), label=label)
        # Held-out (hidden-test) procedures ship only their node list: edges are withheld.
        gt_edges = task_graph_json.get("edges")
        has_ground_truth = gt_edges is not None
        gt_edges = gt_edges or []
        for edge in gt_edges:
            # edge = [dst, src] means src → dst (src is precondition of dst)
            GT.add_edge(edge[1], edge[0])

        n_gt_nodes = len(GT.nodes)  # includes START and END
        n_actual_steps = n_gt_nodes - 2  # excluding START and END

        # Matrix size: beta + GT nodes + gamma
        n_matrix = n_gt_nodes + 2  # 0=beta, 1..n=GTnodes, n+1=gamma

        # Build step names (GT node ID → name)
        step_names = {}
        for node_str, label in task_graph_json["steps"].items():
            node_id = int(node_str)
            step_names[node_id] = label

        # Build taxonomy → GT node ID mapping via description matching
        taxonomy_scenario = taxonomy_json.get(activity_name, {})
        description_to_tax_id = {}
        for tax_id_str, step_info in taxonomy_scenario.items():
            description_to_tax_id[step_info["name"]] = int(tax_id_str)

        # Map GT node → taxonomy ID
        mapping_nodes = {}
        for node in GT.nodes:
            label = GT.nodes[node]["label"]
            if label in description_to_tax_id:
                mapping_nodes[node] = description_to_tax_id[label]

        # Reverse: taxonomy ID → GT node
        id_to_node = {mapping_nodes[node]: node for node in mapping_nodes}

        # Collect training sequences from annotations
        train_sequences = []
        for video, video_data in annotations_json.items():
            if video_data["scenario"].lower().replace(" ", "") != activity_name:
                continue
            seq = []
            for segment in video_data["segments"]:
                step_id = segment["step_id"]
                if step_id not in seq:
                    seq.append(step_id)
            if seq and seq not in train_sequences:
                train_sequences.append(seq)

        if not train_sequences:
            continue

        # Convert sequences to matrix indices following the official DO code:
        # [beta, START, unique steps..., END, gamma]
        # Matrix indices: beta=0, START=1, GT nodes 1..(n-1), END=n, gamma=n+1
        # where n = n_gt_nodes (number of GT nodes including START and END)
        # taxonomy ID → GT node ID → matrix index (+1)
        converted_seqs = []
        for seq in train_sequences:
            new_seq = [0, 1]  # beta, START
            for tax_id in seq:
                gt_node = id_to_node.get(tax_id)
                if gt_node is not None:
                    new_seq.append(gt_node + 1)  # GT node ID + 1 → matrix index
                # else: taxonomy ID not in GT (skip)
            new_seq.append(n_gt_nodes)  # END = n_gt_nodes → matrix index
            new_seq.append(n_matrix - 1)  # gamma = n_matrix - 1 → matrix index
            converted_seqs.append(new_seq)

        # Build GT adjacency matrix
        # GT has nodes 0..(n-1), indexed by GT node ID.
        # Graph orientation: gt_adj[u, v] = 1 means edge u -> v (u is precondition of v).
        # This matches the orientation used by the predicted adjacency matrix
        # and by NetworkX's adjacency_matrix with nodelist=sorted(GT.nodes).
        # Official code: GT.add_edge(edge[1], edge[0]) where edge = [dst, src] -> edge src -> dst
        gt_adj = np.zeros((n_gt_nodes, n_gt_nodes), dtype=np.float32)
        for edge in gt_edges:
            dst, src = edge[0], edge[1]
            if dst < n_gt_nodes and src < n_gt_nodes:
                gt_adj[src, dst] = 1.0  # Graph orientation: src -> dst

        scenario_name = activity_name.replace("_", " ").title()

        results.append(TaskGraphData(
            scenario_name=scenario_name,
            num_steps=n_actual_steps,
            sequences=converted_seqs,
            gt_adj_matrix=gt_adj,
            step_names=step_names,
            gt_node_count=n_gt_nodes,
            has_ground_truth=has_ground_truth,
        ))

    return results


def generate_synthetic_data(
    num_scenarios: int = 3,
    avg_nodes: int = 8,
    avg_sequences: int = 10,
    seed: int = 42
) -> List[TaskGraphData]:
    """
    Generate synthetic task graph data for testing.

    Creates random DAGs and generates sequences that are topological sorts
    of those DAGs. This mimics the structure of CaptainCook4D/EgoPER data.
    """
    np.random.seed(seed)
    results = []

    for s_idx in range(num_scenarios):
        n_gt = avg_nodes + np.random.randint(-2, 3)
        n_gt = max(5, min(n_gt, 15))  # at least 5 GT nodes (START, steps, END)
        n_steps = n_gt - 2  # actual steps
        n_matrix = n_gt + 2  # including beta and gamma

        # Generate random DAG over GT node IDs 1..(n_gt-2) (steps only)
        adj = np.zeros((n_gt, n_gt), dtype=np.float32)

        edge_prob = 0.25
        for i in range(1, n_gt - 1):  # step nodes
            for j in range(1, i):  # can only depend on earlier nodes
                if np.random.random() < edge_prob:
                    adj[i, j] = 1.0  # j → i

        # Ensure connectivity
        for i in range(1, n_gt - 1):
            if np.sum(adj[i, :]) == 0:
                j = np.random.randint(1, i) if i > 1 else 1
                adj[i, j] = 1.0

        # Add START → steps with no preconditions
        for i in range(1, n_gt - 1):
            if np.sum(adj[i, :]) == 0:
                adj[i, 0] = 1.0  # START → i

        # Add steps with no successors → END
        for i in range(1, n_gt - 1):
            if np.sum(adj[:, i]) == 0:
                adj[n_gt - 1, i] = 1.0  # END ← i

        # Generate topological sorts
        sequences = []
        num_seqs = avg_sequences + np.random.randint(-3, 4)
        num_seqs = max(3, min(num_seqs, 20))

        for _ in range(num_seqs):
            seq_gt = _generate_topological_sort(n_gt)
            # seq_gt is a list of GT node IDs for steps (1..n_gt-2)
            # Matrix seq: [0=beta, 1=START, 2..n_gt=GT nodes, n_gt+1=gamma]
            seq_matrix = [0, 1] + seq_gt + [n_gt - 1, n_matrix - 1]
            sequences.append(seq_matrix)

        step_names = {}
        for node_id in range(n_gt):
            if node_id == 0:
                step_names[node_id] = "START"
            elif node_id == n_gt - 1:
                step_names[node_id] = "END"
            else:
                step_names[node_id] = f"Step_{node_id}"

        results.append(TaskGraphData(
            scenario_name=f"Synthetic_Scenario_{s_idx + 1}",
            num_steps=n_steps,
            sequences=sequences,
            gt_adj_matrix=adj,
            step_names=step_names,
            gt_node_count=n_gt,
        ))

    return results


def _generate_topological_sort(n_gt: int) -> List[int]:
    """Generate a random topological sort over GT node IDs 1..(n_gt-2)."""
    step_nodes = list(range(1, n_gt - 1))
    np.random.shuffle(step_nodes)
    return step_nodes


def load_data(repo_data_path: str = None, use_synthetic: bool = False) -> List[TaskGraphData]:
    """
    Load data. If repo_data_path is provided, load CaptainCook4D from the repo.
    Otherwise, generate synthetic data.
    """
    if use_synthetic or repo_data_path is None or not os.path.exists(repo_data_path):
        print("Using synthetic data for testing")
        return generate_synthetic_data(num_scenarios=5, avg_nodes=8, seed=42)
    else:
        print(f"Loading CaptainCook4D from {repo_data_path}")
        return load_captaincook4d(repo_data_path)
