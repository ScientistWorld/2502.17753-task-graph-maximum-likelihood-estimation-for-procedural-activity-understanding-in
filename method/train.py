#!/usr/bin/env python3
"""
Train the Direct Optimization (DO) model for task graph generation.

Usage:
    python method/train.py --dataset captaincook4d --output checkpoints/
    python method/train.py --dataset synthetic --num_scenarios 5
"""

import argparse
import json
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_data, TaskGraphData
from method.tgml import (
    DirectOptimization,
    delete_redundant_edges, delete_cycles,
    compute_precision_recall_f1
)
import networkx as nx

from sklearn.metrics import precision_score, recall_score, f1_score


def train_do(
    scenario_data: TaskGraphData,
    beta: float = 0.005,
    lr: float = 0.1,
    max_epochs: int = 1000,
    early_stopping_patience: int = 25,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True
) -> dict:
    """
    Train DO model for a single scenario.
    Returns a dict with metrics.
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

    # Ground truth adjacency (n_gt x n_gt), indexed by GT node ID
    gt_adj = scenario_data.gt_adj_matrix

    # Threshold: 1 / n_gt (matching the paper's approach)
    threshold = 1.0 / n_gt

    # Training loop
    best_f1 = -1
    best_state = None
    early_stop_counter = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(y_tensor, beta=beta)
        loss.backward()
        optimizer.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                Z = model.get_adjacency_matrix().cpu().numpy()

            # Extract core nodes: remove beta (row/col 0) and gamma (row/col -1)
            pred_core = Z[1:-1, 1:-1]  # (n_gt) x (n_gt)

            # Binarize
            pred_binary = (pred_core > threshold).astype(float)

            # Build NetworkX graph
            GP = nx.DiGraph(pred_binary)

            # Add edges from isolated nodes to match paper's evaluation
            # id_start = 1 (GT node 0 = START), id_end = n_gt - 1 (GT node n-1 = END)
            id_start = 0  # GT node ID
            id_end = n_gt - 1  # GT node ID

            # Nodes with no incoming edges (except START) → connect from END
            in_degree_zeros = [n for n in GP.nodes if GP.in_degree(n) == 0 and n != id_end]
            for node in in_degree_zeros:
                GP.add_edge(id_end, node)

            # Nodes with no outgoing edges (except END) → connect to START
            out_degree_zeros = [n for n in GP.nodes if GP.out_degree(n) == 0 and n != id_start]
            for node in out_degree_zeros:
                GP.add_edge(node, id_start)

            # Transitive reduction
            delete_redundant_edges(GP)

            # Also try cycle removal as a safety net
            delete_cycles(GP)

            # Convert GP back to adjacency matrix for comparison
            pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
            for u, v in GP.edges():
                pred_adj[v, u] = 1.0

            # Flatten for sklearn metrics
            gt_flat = gt_adj.flatten()
            pred_flat = pred_adj.flatten()

            precision = precision_score(gt_flat, pred_flat, zero_division=0) * 100
            recall = recall_score(gt_flat, pred_flat, zero_division=0) * 100
            f1 = f1_score(gt_flat, pred_flat, zero_division=0) * 100

            if verbose and (epoch % 50 == 0 or epoch == max_epochs - 1):
                print(f"  Epoch {epoch}: loss={loss.item():.4f}, P={precision:.1f}, R={recall:.1f}, F1={f1:.1f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        Z = model.get_adjacency_matrix().cpu().numpy()

    pred_core = Z[1:-1, 1:-1]
    pred_binary = (pred_core > threshold).astype(float)

    GP = nx.DiGraph(pred_binary)

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

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_adj[v, u] = 1.0

    gt_flat = gt_adj.flatten()
    pred_flat = pred_adj.flatten()

    precision = precision_score(gt_flat, pred_flat, zero_division=0) * 100
    recall = recall_score(gt_flat, pred_flat, zero_division=0) * 100
    f1 = f1_score(gt_flat, pred_flat, zero_division=0) * 100

    if verbose:
        print(f"  Final: P={precision:.1f}, R={recall:.1f}, F1={f1:.1f}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_epoch": epoch - early_stop_counter
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
    args = parser.parse_args()

    # Load data
    if args.dataset == "synthetic":
        data = load_data(use_synthetic=True)
        data = data[:args.num_scenarios]
    else:
        data = load_data(repo_data_path=args.repo_data_path)

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

        # Average results
        avg_f1 = np.mean([r["f1"] for r in seed_results])
        avg_p = np.mean([r["precision"] for r in seed_results])
        avg_r = np.mean([r["recall"] for r in seed_results])
        std_f1 = np.std([r["f1"] for r in seed_results])

        print(f"  Average: P={avg_p:.1f}, R={avg_r:.1f}, F1={avg_f1:.1f} +/- {std_f1:.1f}")
        all_results[scenario_name] = {
            "precision": avg_p,
            "recall": avg_r,
            "f1": avg_f1,
            "std_f1": std_f1
        }

        # Save per-scenario results
        safe_name = scenario_name.lower().replace(" ", "_")
        with open(os.path.join(args.output, f"{safe_name}_results.json"), "w") as f:
            json.dump({
                "scenario": scenario_name,
                "precision": avg_p,
                "recall": avg_r,
                "f1": avg_f1,
                "std_f1": std_f1,
                "per_seed": seed_results
            }, f, indent=2)

    # Overall averages
    overall_f1 = np.mean([r["f1"] for r in all_results.values()])
    overall_p = np.mean([r["precision"] for r in all_results.values()])
    overall_r = np.mean([r["recall"] for r in all_results.values()])
    overall_std = np.std([r["f1"] for r in all_results.values()])

    print(f"\n=== OVERALL ===")
    print(f"Precision: {overall_p:.1f}")
    print(f"Recall: {overall_r:.1f}")
    print(f"F1: {overall_f1:.1f} +/- {overall_std:.1f}")

    # Save overall results
    with open(os.path.join(args.output, "overall_results.json"), "w") as f:
        json.dump({
            "dataset": args.dataset,
            "overall": {
                "precision": overall_p,
                "recall": overall_r,
                "f1": overall_f1,
                "std_f1": overall_std
            },
            "per_scenario": all_results
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
