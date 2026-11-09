#!/usr/bin/env python3
"""
Train the paper's Direct Optimization (DO) model with the TGML loss.

Faithful to the paper (Sections III-A/B, Appendix A-B/A-C) and to the released
code (task-graph-learning/DO/train_with_gt.py):
  * one learnable score matrix per procedure, masked row-softmax Z = softmax(A);
  * TGML loss (Eq. 11), beta = 0.005, Adam with lr = 0.1, at most 1000 epochs;
  * early stopping on the weakly supervised Sequence Accuracy (SA, Eq. 25): after
    every epoch the current Z is binarised at 1/n, completed with START/END edges
    and transitively reduced; if the graph is a DAG its SA on the demonstrations
    is computed. Once SA > 0.95, training halts after 25 consecutive epochs
    without SA improvement (a repeated SA of 1.0 counts as no improvement). The
    model state at the halting epoch is the output.
  * beta = 0.005, raised to 0.5 for a procedure none of whose demonstrations
    contains every key-step (paper Appendix A-C: "we found it advantageous to
    increase it to 0.50 in cases where sequences do not include all
    taxonomy-defined steps"); never triggered on CaptainCook4D.
Ground-truth graphs are never read: only the demonstrations and the node list.
Datasets: CaptainCook4D (24 procedures) and the noisy-demonstration benchmark
(data/noisy_demos, public and held-out procedures).

Output: <output>/<procedure>_predictions.npz with ``pred_cores`` (n_seeds, n, n),
the START..END block of Z for every seed, plus ``seeds``; and
<output>/<procedure>_training.json with per-seed stopping metadata.

Usage:
    python method/train.py --output checkpoints/do
    python method/train.py --output /tmp/x --seeds 42 --procedures coffee
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import TaskGraphData, load_data, load_noisy_demos
from method.tgml import (
    DirectOptimization,
    build_tgml_events,
    delete_redundant_edges,
    task_graph_maximum_likelihood_loss,
    task_graph_maximum_likelihood_loss_batched,
)

DEFAULT_DATA = "/home/user/data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d"
DEFAULT_SEEDS = [42, 1337, 2024, 2025, 2026]
EPS = 1e-6


def extract_predecessors(graph: nx.DiGraph) -> Dict[int, List[int]]:
    """Released utility: node -> predicted preconditions (graph successors)."""
    return {node: list(graph.successors(node)) for node in graph.nodes}


def sequences_accuracy(sequences: List[List[int]], pred_predecessors: Dict[int, List[int]]) -> float:
    """Sequence Accuracy (paper Eq. 25-26, released implementation)."""
    accuracy = 0.0
    for sequence in sequences:
        correct = 0.0
        for i in range(len(sequence)):
            preds = pred_predecessors[sequence[i]]
            if len(sequence[:i]) == 0 and len(preds) == 0:
                correct += 1.0
            elif len(sequence[:i]) > 0 and len(preds) > 0:
                correct += len(set(sequence[:i]).intersection(set(preds))) / len(preds)
        accuracy += correct / len(sequence)
    return accuracy / len(sequences) if len(sequences) > 0 else 0.0


def released_sequence_accuracy(pred_core: np.ndarray, n_gt: int, sa_sequences: List[List[int]]) -> float:
    """SA of the current graph as computed in the released training loop (0 if cyclic)."""
    binary = np.where(pred_core < (1.0 / n_gt), 0, 1)
    graph = nx.DiGraph(binary)
    id_start, id_end = 0, n_gt - 1
    for node in [v for v in graph.nodes if graph.in_degree(v) == 0 and v != id_end and v != id_start]:
        graph.add_edge(id_end, node)
    for node in [v for v in graph.nodes if graph.out_degree(v) == 0 and v != id_end and v != id_start]:
        graph.add_edge(node, id_start)
    delete_redundant_edges(graph)
    if not nx.is_directed_acyclic_graph(graph):
        return 0.0
    return sequences_accuracy(sa_sequences, extract_predecessors(graph))


def sequence_accuracy_context(sa_sequences: List[List[int]], n_gt: int):
    """Precompute, per demonstration, which nodes precede each position."""
    context = []
    for seq in sa_sequences:
        before = np.zeros((len(seq), n_gt), dtype=bool)
        for t in range(1, len(seq)):
            before[t] = before[t - 1]
            before[t, seq[t - 1]] = True
        context.append((np.asarray(seq, dtype=np.int64), before))
    return context


def fast_released_sequence_accuracy(pred_core: np.ndarray, n_gt: int, context) -> float:
    """Vectorised ``released_sequence_accuracy`` (same graph semantics, same
    summation order). For a DAG the sequential redundant-edge deletion of the
    released code equals the unique transitive reduction; a cyclic graph scores 0."""
    adj = pred_core >= (1.0 / n_gt)
    id_start, id_end = 0, n_gt - 1
    nodes = np.arange(n_gt)
    boundary = (nodes != id_start) & (nodes != id_end)
    adj[id_end, np.nonzero(~adj.any(axis=0) & boundary)[0]] = True
    adj[np.nonzero(~adj.any(axis=1) & boundary)[0], id_start] = True
    np.fill_diagonal(adj, False)  # self-loops are always deleted as redundant
    reach = adj.copy()
    for k in range(n_gt):
        reach |= np.outer(reach[:, k], reach[k, :])
    if reach.diagonal().any():
        return 0.0
    reduced = adj & ~((adj.astype(np.int64) @ reach.astype(np.int64)) > 0)
    accuracy = 0
    for seq, before in context:
        preds = reduced[seq]
        sizes = preds.sum(axis=1)
        counts = (preds & before).sum(axis=1)
        correct = 0
        for t in range(len(seq)):
            if t == 0:
                if sizes[0] == 0:
                    correct += 1
            elif sizes[t] > 0:
                correct += counts[t] / sizes[t]
        accuracy += correct / len(seq)
    return float(accuracy / len(context)) if context else 0.0


def paper_beta(scenario_data: TaskGraphData) -> float:
    """beta = 0.005, or 0.5 when no demonstration contains every key-step."""
    steps = set(range(2, scenario_data.gt_node_count))  # matrix indices of key-steps
    complete = any(steps <= set(seq) for seq in scenario_data.sequences)
    return 0.005 if complete else 0.5


def _padded_sequences(sequences: List[List[int]]) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    y = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        y[i, :len(seq)] = seq
    return torch.tensor(y, dtype=torch.long)


SA_CROSSCHECK_EPOCHS = (0, 1, 5, 20, 60, 150, 400)


def train_do(
    scenario_data: TaskGraphData,
    beta: Optional[float] = None,
    lr: float = 0.1,
    max_epochs: int = 1000,
    seed: int = 42,
    early_stopping: bool = True,
    patience: int = 25,
    sa_threshold: float = 0.95,
    verify_loss: bool = True,
    verbose: bool = False,
) -> dict:
    """Train DO on one procedure. Returns the halting-state core and metadata.

    Also returns (for analysis only) the core after the first update and the
    core at the best-SA epoch; the paper's output is ``pred_core``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if beta is None:
        beta = paper_beta(scenario_data)

    n_gt = scenario_data.gt_node_count
    n_matrix = n_gt + 2
    sequences = scenario_data.sequences
    if not sequences:
        raise ValueError(f"{scenario_data.scenario_name}: no demonstrations")

    model = DirectOptimization(n_matrix, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    events = build_tgml_events(sequences, n_matrix)
    # SA uses the demonstrations as node-id sequences with START and END.
    sa_sequences = [[step - 1 for step in seq[1:-1]] for seq in sequences]
    sa_context = sequence_accuracy_context(sa_sequences, n_gt)
    use_fast_sa = True

    if verify_loss:
        with torch.no_grad():
            z0 = model.get_adjacency_matrix() + EPS
            reference_loss = float(task_graph_maximum_likelihood_loss(_padded_sequences(sequences), z0, beta))
            batched_loss = float(task_graph_maximum_likelihood_loss_batched(z0, events, beta))
        if abs(reference_loss - batched_loss) > 1e-4 * max(1.0, abs(reference_loss)):
            raise RuntimeError(f"batched TGML loss mismatch: {reference_loss} vs {batched_loss}")

    best_sa = -float("inf")
    best_core: Optional[np.ndarray] = None
    best_epoch = 0
    counter = 0
    stop_epoch = max_epochs
    first_core: Optional[np.ndarray] = None
    last_sa = 0.0
    loss_value = float("nan")

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = task_graph_maximum_likelihood_loss_batched(model.get_adjacency_matrix() + EPS, events, beta)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())

        model.eval()
        with torch.no_grad():
            core = model.get_adjacency_matrix()[1:-1, 1:-1].cpu().numpy().copy()
        if epoch == 0:
            first_core = core.copy()

        if early_stopping:
            if use_fast_sa:
                sa = fast_released_sequence_accuracy(core, n_gt, sa_context)
                if epoch in SA_CROSSCHECK_EPOCHS:
                    slow = released_sequence_accuracy(core, n_gt, sa_sequences)
                    if slow != sa:
                        print(f"    WARNING: fast SA {sa!r} != released SA {slow!r} "
                              f"({scenario_data.scenario_name}, seed {seed}, epoch {epoch}); using released SA")
                        use_fast_sa = False
                        sa = slow
            else:
                sa = released_sequence_accuracy(core, n_gt, sa_sequences)
            last_sa = sa
            if sa > sa_threshold:
                if sa == 1.0 and sa == best_sa:
                    counter += 1
                elif sa >= best_sa:
                    best_sa = sa
                    best_core = core.copy()
                    best_epoch = epoch + 1
                    counter = 0
                else:
                    counter += 1
            if verbose and epoch % 50 == 0:
                print(f"    epoch {epoch + 1}: loss={loss_value:.4f} SA={sa:.3f}")
            if counter == patience:
                stop_epoch = epoch + 1
                break

    with torch.no_grad():
        final_core = model.get_adjacency_matrix()[1:-1, 1:-1].cpu().numpy().copy()

    return {
        "pred_core": final_core.astype(np.float32),
        "first_core": first_core.astype(np.float32),
        "best_core": (best_core if best_core is not None else final_core).astype(np.float32),
        "stop_epoch": int(stop_epoch),
        "best_epoch": int(best_epoch),
        "best_sequence_accuracy": float(best_sa) if best_core is not None else None,
        "final_sequence_accuracy": float(last_sa),
        "final_loss": loss_value,
        "seed": int(seed),
        "beta": float(beta),
        "fast_sa_consistent": bool(use_fast_sa),
    }


def procedure_key(scenario_data: TaskGraphData) -> str:
    return scenario_data.scenario_name.lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DO model (TGML loss) on CaptainCook4D")
    parser.add_argument("--repo_data_path", type=str, default=DEFAULT_DATA)
    parser.add_argument("--output", type=str, default="checkpoints/do")
    parser.add_argument("--beta", type=float, default=None,
                        help="Fixed beta; default: the paper's rule (0.005, or 0.5 without complete demonstrations).")
    parser.add_argument("--datasets", nargs="+", default=["captaincook4d", "noisy_demos"],
                        choices=["captaincook4d", "noisy_demos"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--no_early_stopping", action="store_true",
                        help="Train for max_epochs and output the final state (not the paper's protocol).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--procedures", type=str, nargs="*", default=None,
                        help="Optional procedure keys (e.g. coffee ramen); default: all 24.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    data = []
    if "captaincook4d" in args.datasets:
        data += load_data(repo_data_path=args.repo_data_path)
    if "noisy_demos" in args.datasets:
        data += load_noisy_demos("all")
    if args.procedures:
        wanted = {p.lower() for p in args.procedures}
        data = [d for d in data if procedure_key(d) in wanted]
    os.makedirs(args.output, exist_ok=True)
    print(f"Training DO on {len(data)} procedures, seeds={args.seeds}, beta={args.beta or 'paper rule'}, "
          f"early_stopping={not args.no_early_stopping}")

    for scenario_data in data:
        key = procedure_key(scenario_data)
        runs = [
            train_do(scenario_data, beta=args.beta, lr=args.lr, max_epochs=args.max_epochs, seed=seed,
                     early_stopping=not args.no_early_stopping, patience=args.patience,
                     verbose=args.verbose)
            for seed in args.seeds
        ]
        np.savez_compressed(
            os.path.join(args.output, f"{key}_predictions.npz"),
            pred_cores=np.stack([r["pred_core"] for r in runs], axis=0),
            seeds=np.array(args.seeds, dtype=np.int64),
        )
        with open(os.path.join(args.output, f"{key}_training.json"), "w") as f:
            json.dump({"procedure": scenario_data.scenario_name,
                       "runs": [{k: r[k] for k in ("seed", "beta", "stop_epoch", "best_epoch", "best_sequence_accuracy",
                                                   "final_sequence_accuracy", "final_loss", "fast_sa_consistent")}
                                for r in runs]},
                      f, indent=2)
        print(f"  {scenario_data.scenario_name}: beta {runs[0]['beta']}, stop epochs {[r['stop_epoch'] for r in runs]}, "
              f"SA {[round(r['final_sequence_accuracy'], 3) for r in runs]}", flush=True)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
