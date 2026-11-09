"""Exact batched TGML algebra, training variants, and fresh seed reruns.

Uses only the available method, data, and public training evaluator. No test
membership or scores are used in training or checkpoint selection.
"""
import copy
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from data.loader import load_captaincook4d
from method.tgml import DirectOptimization, task_graph_maximum_likelihood_loss
from eval.train.evaluate import post_process_pred_adj

ROOT = Path('/home/user')
OUT = ROOT / 'redteam'
DATA = ROOT / 'data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d'
SEEDS = [7, 19, 31]


def event_masks(sequences, n):
    observed, remaining, current = [], [], []
    for sequence in sequences:
        assert len(sequence) == len(set(sequence)), 'Batching requires unique loader sequences'
        for i in range(1, len(sequence) - 1):
            before = set(sequence[:i]) & set(range(1, n - 1))
            after = set(range(1, n - 1)) - before
            if not before or not after:
                continue
            observed.append([int(j in before) for j in range(n)])
            remaining.append([int(j in after) for j in range(n)])
            current.append(sequence[i])
    return (torch.tensor(observed, dtype=torch.float32),
            torch.tensor(remaining, dtype=torch.float32), torch.tensor(current))


def loss_batched(model, masks):
    before, after, current = masks
    z = model.get_adjacency_matrix() + 1e-6
    numerator = (z[current] * before).sum(dim=1)
    denominator = ((z @ before.T).T * after).sum(dim=1)
    return (0.005 * denominator.log() - numerator.log()).sum()


def accuracy(core, sequences, n):
    # Released sequence-accuracy selection: graph must be acyclic before
    # cycle deletion, with the same boundary completion and reduction.
    from eval.train.evaluate import add_boundary_edges, delete_redundant_edges
    graph = nx.DiGraph((core >= 1.0 / n).astype(float))
    add_boundary_edges(graph, 0, n - 1)
    delete_redundant_edges(graph)
    if not nx.is_directed_acyclic_graph(graph):
        return 0.0
    scores = []
    for sequence in sequences:
        seq = [x - 1 for x in sequence if 1 <= x <= n]
        correct = 0.0
        for i, node in enumerate(seq):
            previous = set(seq[:i])
            parents = set(graph.successors(node))
            if not previous and not parents:
                correct += 1
            elif previous and parents:
                correct += len(previous & parents) / len(parents)
        scores.append(correct / len(seq))
    return float(np.mean(scores))


def save(folder, name, cores, seeds):
    folder = OUT / folder / 'predictions'
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f'{name.lower()}_predictions.npz',
                        pred_cores=np.stack(cores), seeds=np.array(seeds))


def main():
    torch.set_num_threads(1)
    data = load_captaincook4d(str(DATA))
    started = time.time()
    proof = []
    metadata = {}
    for sd in data:
        paths = [OUT / x / 'predictions' / f'{sd.scenario_name.lower()}_predictions.npz'
                 for x in ['reference_rerun', 'better_more_training', 'better_sequence_selection', 'weaker_one_epoch']]
        if all(p.exists() for p in paths):
            continue
        predictions = {x: [] for x in ['reference_rerun', 'better_more_training', 'better_sequence_selection', 'weaker_one_epoch']}
        metadata[sd.scenario_name] = []
        n = sd.gt_node_count + 2
        masks = event_masks(sd.sequences, n)
        for seed in SEEDS:
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = DirectOptimization(n)
            opt = torch.optim.Adam(model.parameters(), lr=0.1)
            # Compare to the actual implementation before optimizing.
            if seed == SEEDS[0]:
                maxlen = max(map(len, sd.sequences))
                y = torch.zeros((len(sd.sequences), maxlen), dtype=torch.long)
                for i, seq in enumerate(sd.sequences):
                    y[i, :len(seq)] = torch.tensor(seq)
                slow = model(y)
                gslow = torch.autograd.grad(slow, model.adjacency_matrix.task_graph)[0]
                fast = loss_batched(model, masks)
                gfast = torch.autograd.grad(fast, model.adjacency_matrix.task_graph)[0]
                err = float((gslow - gfast).abs().max())
                assert err < 1e-4, (sd.scenario_name, err)
                proof.append({'scenario': sd.scenario_name, 'loss_original': float(slow.detach()),
                              'loss_batched': float(fast.detach()), 'gradient_max_error': err})
            best_acc, best_core, best_epoch, patience = -1, None, -1, 0
            selection_done = False
            for epoch in range(3000):
                opt.zero_grad()
                loss = loss_batched(model, masks)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    core = model.get_adjacency_matrix().numpy()[1:-1, 1:-1].copy()
                if epoch == 0:
                    predictions['weaker_one_epoch'].append(core.copy())
                if epoch == 999:
                    predictions['reference_rerun'].append(core.copy())
                if epoch < 1000 and not selection_done:
                    acc = accuracy(core, sd.sequences, sd.gt_node_count)
                    if acc > 0.95:
                        if acc == 1.0 and acc == best_acc:
                            patience += 1
                        elif acc >= best_acc:
                            best_acc, best_core, best_epoch, patience = acc, core.copy(), epoch, 0
                        else:
                            patience += 1
                    if patience >= 25:
                        selection_done = True
                if epoch == 999:
                    predictions['better_sequence_selection'].append(
                        best_core.copy() if best_core is not None else core.copy())
            predictions['better_more_training'].append(core.copy())
            metadata[sd.scenario_name].append({'seed': seed, 'sequence_accuracy': best_acc,
                                              'selected_epoch': best_epoch + 1})
            print(sd.scenario_name, seed, 'seconds', round(time.time() - started, 1), flush=True)
        for label, cores in predictions.items():
            save(label, sd.scenario_name, cores, SEEDS)
        (OUT / 'training_metadata.json').write_text(json.dumps(metadata, indent=2))
        (OUT / 'algebra_verification.json').write_text(json.dumps(proof, indent=2))

    # An independent scalar-loop reference run verifies final graph agreement
    # after all 1000 Adam updates on the smallest procedure.
    sd = min(data, key=lambda x: x.gt_node_count)
    torch.manual_seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    model = DirectOptimization(sd.gt_node_count + 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    y = torch.zeros((len(sd.sequences), max(map(len, sd.sequences))), dtype=torch.long)
    for i, seq in enumerate(sd.sequences):
        y[i, :len(seq)] = torch.tensor(seq)
    for _ in range(1000):
        opt.zero_grad()
        loss = model(y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scalar_core = model.get_adjacency_matrix().numpy()[1:-1, 1:-1]
    batched = np.load(OUT / 'reference_rerun/predictions' / f'{sd.scenario_name.lower()}_predictions.npz')['pred_cores'][0]
    same = np.array_equal(post_process_pred_adj(scalar_core, sd.gt_node_count),
                          post_process_pred_adj(batched, sd.gt_node_count))
    (OUT / 'scalar_verification.json').write_text(json.dumps({
        'scenario': sd.scenario_name, 'seed': SEEDS[0], 'epochs': 1000,
        'max_core_difference': float(np.max(np.abs(scalar_core - batched))),
        'same_evaluated_graph': bool(same)}, indent=2))
    assert same


if __name__ == '__main__':
    main()
