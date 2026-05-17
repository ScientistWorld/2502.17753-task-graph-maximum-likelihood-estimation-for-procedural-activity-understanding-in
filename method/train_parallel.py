#!/usr/bin/env python3
"""
Parallel training runner for DO on all 24 CaptainCook4D scenarios.
Trains each scenario in a separate process for parallelism.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_data, TaskGraphData
from method.tgml import DirectOptimization


def extract_predecessors(graph):
    predecessors = {}
    for current_node in graph.nodes:
        predecessors[current_node] = list(graph.successors(current_node))
    return predecessors


def sequences_accuracy(sequences, pred_predecessors):
    accuracy = 0.0
    for sequence in sequences:
        correct = 0.0
        for i in range(len(sequence)):
            if len(sequence[:i]) == 0 and len(pred_predecessors[sequence[i]]) == 0:
                correct += 1.0
            elif len(sequence[:i]) > 0 and len(pred_predecessors[sequence[i]]) > 0:
                correct += (
                    len(set(sequence[:i]).intersection(set(pred_predecessors[sequence[i]])))
                    / len(pred_predecessors[sequence[i]])
                )
        accuracy += correct / len(sequence)
    return accuracy / len(sequences) if len(sequences) > 0 else 0.0


def post_process_prediction(pred_core, n_gt, threshold):
    import networkx as nx
    from method.tgml import delete_redundant_edges, delete_cycles

    pred_binary = (pred_core > threshold).astype(float)
    graph = nx.DiGraph(pred_binary)
    id_start = 0
    id_end = n_gt - 1

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

    delete_redundant_edges(graph)
    delete_cycles(graph)

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in graph.edges():
        pred_adj[v, u] = 1.0
    return pred_adj


def train_single_scenario(args):
    """Train DO on a single scenario. Designed to run in a subprocess."""
    scenario_data, beta, lr, max_epochs, seed, device, output_dir = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_gt = scenario_data.gt_node_count
    n_matrix = n_gt + 2

    sequences = scenario_data.sequences
    if not sequences:
        return None

    max_len = max(len(seq) for seq in sequences)
    y = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        y[i, :len(seq)] = seq
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    model = DirectOptimization(n_matrix, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_seq_no_beta_gamma = np.array(
        [[0] + [step - 1 for step in seq[2:-2]] + [n_gt - 1] for seq in sequences],
        dtype=np.int64,
    )
    threshold = 1.0 / (n_matrix - 2)

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(y_tensor, beta=beta)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        Z = model.get_adjacency_matrix().cpu().numpy()
    pred_core = Z[1:-1, 1:-1]

    safe_name = scenario_data.scenario_name.lower().replace(" ", "_")
    np.savez_compressed(
        os.path.join(output_dir, f"{safe_name}_predictions.npz"),
        pred_cores=np.array([pred_core], dtype=np.float32),
        seeds=np.array([seed], dtype=np.int64),
    )

    return {
        "scenario": scenario_data.scenario_name,
        "seed": seed,
        "pred_core_shape": pred_core.shape,
    }


def train_scenario_cpu(scenario_data, beta, lr, max_epochs, seed, output_dir):
    """Train one scenario on CPU (no GPU CUDA issues)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_gt = scenario_data.gt_node_count
    n_matrix = n_gt + 2

    sequences = scenario_data.sequences
    if not sequences:
        return None

    max_len = max(len(seq) for seq in sequences)
    y = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        y[i, :len(seq)] = seq
    y_tensor = torch.tensor(y, dtype=torch.long, device='cpu')

    model = DirectOptimization(n_matrix, device='cpu').to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(y_tensor, beta=beta)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        Z = model.get_adjacency_matrix().cpu().numpy()
    pred_core = Z[1:-1, 1:-1]

    safe_name = scenario_data.scenario_name.lower().replace(" ", "_")
    np.savez_compressed(
        os.path.join(output_dir, f"{safe_name}_predictions.npz"),
        pred_cores=np.array([pred_core], dtype=np.float32),
        seeds=np.array([seed], dtype=np.int64),
    )

    return {
        "scenario": scenario_data.scenario_name,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel DO training for CaptainCook4D")
    parser.add_argument("--repo_data_path", type=str,
                        default="/home/user/data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d")
    parser.add_argument("--output", type=str, default="checkpoints/do_captaincook4d")
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    data = load_data(repo_data_path=args.repo_data_path)
    print(f"Loaded {len(data)} scenarios, training on CPU with "
          f"{args.num_workers or cpu_count()} workers")

    os.makedirs(args.output, exist_ok=True)

    # Remove old results
    import glob
    for f in glob.glob(os.path.join(args.output, "*_predictions.npz")):
        os.remove(f)
    for f in glob.glob(os.path.join(args.output, "*_results.json")):
        os.remove(f)

    # Train all scenarios in parallel
    num_workers = args.num_workers or cpu_count()
    print(f"Starting parallel training on {num_workers} workers...")

    t0 = time.time()
    with Pool(num_workers) as pool:
        tasks = [
            (d, args.beta, args.lr, args.max_epochs, args.seed, args.output)
            for d in data
        ]
        results = pool.starmap(train_scenario_cpu, [
            (d, args.beta, args.lr, args.max_epochs, args.seed, args.output)
            for d in data
        ])

    elapsed = time.time() - t0
    print(f"\nAll scenarios trained in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    successful = [r for r in results if r is not None]
    print(f"Successful: {len(successful)}/{len(data)}")

    # Save overall results
    overall = {
        "dataset": "captaincook4d",
        "elapsed_seconds": elapsed,
        "num_scenarios": len(data),
        "num_successful": len(successful),
        "max_epochs": args.max_epochs,
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "overall_results.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
