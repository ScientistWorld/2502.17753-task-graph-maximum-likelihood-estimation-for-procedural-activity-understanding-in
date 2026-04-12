#!/usr/bin/env python3
"""
Train the Direct Optimization (DO) model for task graph generation.

Usage:
    python method/train.py --dataset captaincook4d --output checkpoints/
    python method/train.py --dataset synthetic --num_scenarios 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_data, TaskGraphData
from method.tgml import (
    DirectOptimization,
    delete_redundant_edges,
)


def extract_predecessors(graph: nx.DiGraph) -> dict[int, list[int]]:
    """Match the released utility: node -> list of predecessor nodes."""
    predecessors = {}
    for current_node in graph.nodes:
        predecessors[current_node] = list(graph.successors(current_node))
    return predecessors


def sequences_accuracy(sequences: np.ndarray, pred_predecessors: dict[int, list[int]]) -> float:
    """Score how well a predicted graph explains the observed sequences."""
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


def post_process_prediction(pred_core: np.ndarray, n_gt: int, threshold: float) -> np.ndarray:
    """Mirror the released DO training-time post-processing used before scoring."""
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

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in graph.edges():
        pred_adj[v, u] = 1.0
    return pred_adj


def train_do(
    scenario_data: TaskGraphData,
    beta: float = 0.005,
    lr: float = 0.1,
    max_epochs: int = 1000,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True
) -> dict:
    """
    Train DO model for a single scenario.
    Returns a dict with saved prediction artifacts and weakly supervised metadata.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Matrix size: beta(0) + GT_nodes(1..n) + gamma(n+1)
    n_gt = scenario_data.gt_node_count
    n_matrix = n_gt + 2

    sequences = scenario_data.sequences
    if not sequences:
        return {"error": "No sequences"}

    # Convert sequences to tensor
    max_len = max(len(seq) for seq in sequences)
    y = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        y[i, :len(seq)] = seq

    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    # Create DO model
    model = DirectOptimization(n_matrix, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Official DO early stopping uses sequence accuracy on sequences without beta/gamma.
    train_seq_no_beta_gamma = np.array(
        [[0] + [step - 1 for step in seq[2:-2]] + [n_gt - 1] for seq in sequences],
        dtype=np.int64,
    )

    # Threshold: 1 / (num_nodes - 2) matching the official code (num_nodes includes beta and gamma)
    threshold = 1.0 / (n_matrix - 2)

    # Training loop. We intentionally keep the released DO behavior:
    # the final epoch state is the output, rather than restoring a custom
    # checkpoint selected with local heuristics.
    last_logged_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(y_tensor, beta=beta)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            Z = model.get_adjacency_matrix().cpu().numpy()

        # Extract core nodes: remove beta (row/col 0) and gamma (row/col -1)
        pred_core = Z[1:-1, 1:-1]  # (n_gt) x (n_gt)

        pred_adj = post_process_prediction(pred_core, n_gt, threshold)
        pred_graph = nx.DiGraph(pred_adj.T)

        sequence_accuracy = 0.0
        if nx.is_directed_acyclic_graph(pred_graph):
            sequence_accuracy = sequences_accuracy(
                train_seq_no_beta_gamma,
                extract_predecessors(pred_graph),
            )

        last_logged_epoch = epoch
        if verbose and (epoch % 50 == 0 or epoch == max_epochs - 1):
            print(
                f"  Epoch {epoch}: loss={loss.item():.4f}, "
                f"seq_acc={sequence_accuracy:.3f}"
            )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        Z = model.get_adjacency_matrix().cpu().numpy()

    pred_core = Z[1:-1, 1:-1]
    pred_adj = post_process_prediction(pred_core, n_gt, threshold)
    pred_graph = nx.DiGraph(pred_adj.T)

    final_sequence_accuracy = 0.0
    if nx.is_directed_acyclic_graph(pred_graph):
        final_sequence_accuracy = sequences_accuracy(
            train_seq_no_beta_gamma,
            extract_predecessors(pred_graph),
        )

    if verbose:
        print(
            f"  Final: seq_acc={final_sequence_accuracy:.3f}"
        )

    return {
        "pred_core": pred_core.astype(np.float32),
        "best_epoch": last_logged_epoch,
        "sequence_accuracy": float(final_sequence_accuracy),
    }


def main():
    parser = argparse.ArgumentParser(description="Train DO model for task graph learning")
    parser.add_argument("--dataset", type=str, default="captaincook4d",
                        choices=["captaincook4d", "synthetic"],
                        help="Dataset to use")
    parser.add_argument("--repo_data_path", type=str,
                        default="/home/user/method/repo/data/captaincook4d",
                        help="Path to repo data")
    parser.add_argument("--output", type=str, default="checkpoints/",
                        help="Output directory for checkpoints")
    parser.add_argument("--beta", type=float, default=0.005,
                        help="Beta parameter for TGML loss")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=1000,
                        help="Maximum epochs per scenario")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    parser.add_argument("--num_scenarios", type=int, default=5,
                        help="Number of synthetic scenarios")
    parser.add_argument("--scenario_filter", type=str, default=None,
                        help="Optional substring filter for scenario names")
    args = parser.parse_args()

    # Load data
    if args.dataset == "synthetic":
        data = load_data(use_synthetic=True)
        data = data[:args.num_scenarios]
    else:
        data = load_data(repo_data_path=args.repo_data_path)

    if args.scenario_filter:
        needle = args.scenario_filter.lower()
        data = [d for d in data if needle in d.scenario_name.lower()]

    print(f"Loaded {len(data)} scenarios")
    for d in data:
        print(f"  {d.scenario_name}: {d.num_steps} steps, {len(d.sequences)} sequences, {d.gt_node_count} GT nodes")

    os.makedirs(args.output, exist_ok=True)

    all_results = {}

    for scenario_data in data:
        scenario_name = scenario_data.scenario_name
        print(f"\nTraining on {scenario_name}...")

        seed_results = []
        for seed in args.seeds:
            print(f"  Seed {seed}:")
            result = train_do(
                scenario_data,
                beta=args.beta,
                lr=args.lr,
                max_epochs=args.max_epochs,
                seed=seed,
                device=args.device,
                verbose=args.verbose
            )
            result["seed"] = seed
            seed_results.append(result)

        avg_sequence_accuracy = np.mean([r["sequence_accuracy"] for r in seed_results])
        std_sequence_accuracy = np.std([r["sequence_accuracy"] for r in seed_results])

        print(
            f"  Average weakly-supervised seq_acc={avg_sequence_accuracy:.3f} "
            f"+/- {std_sequence_accuracy:.3f}"
        )
        all_results[scenario_name] = {
            "sequence_accuracy": avg_sequence_accuracy,
            "std_sequence_accuracy": std_sequence_accuracy
        }

        # Save raw DO predictions for method-agnostic evaluation in eval/.
        safe_name = scenario_name.lower().replace(" ", "_")
        pred_cores = np.stack([r["pred_core"] for r in seed_results], axis=0)
        np.savez_compressed(
            os.path.join(args.output, f"{safe_name}_predictions.npz"),
            pred_cores=pred_cores,
            seeds=np.array([r["seed"] for r in seed_results], dtype=np.int64),
        )

        # Save per-scenario weak supervision metadata only.
        with open(os.path.join(args.output, f"{safe_name}_results.json"), "w") as f:
            json.dump({
                "scenario": scenario_name,
                "sequence_accuracy": avg_sequence_accuracy,
                "std_sequence_accuracy": std_sequence_accuracy,
                "per_seed": [
                    {
                        "seed": r["seed"],
                        "best_epoch": r["best_epoch"],
                        "sequence_accuracy": r["sequence_accuracy"],
                    }
                    for r in seed_results
                ]
            }, f, indent=2)

    overall_sequence_accuracy = np.mean([r["sequence_accuracy"] for r in all_results.values()])
    overall_std = np.std([r["sequence_accuracy"] for r in all_results.values()])

    print(f"\n=== OVERALL ===")
    print(f"Sequence accuracy: {overall_sequence_accuracy:.3f} +/- {overall_std:.3f}")

    # Save overall weak supervision metadata only.
    with open(os.path.join(args.output, "overall_results.json"), "w") as f:
        json.dump({
            "dataset": args.dataset,
            "overall": {
                "sequence_accuracy": overall_sequence_accuracy,
                "std_sequence_accuracy": overall_std
            },
            "per_scenario": all_results
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
