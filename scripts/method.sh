#!/bin/bash
# Run the paper's proposed DO method on CaptainCook4D.
# Supports multi-seed training with per-epoch F1 evaluation and best-model selection.
#
# Usage:
#   bash scripts/method.sh                    # Default: 3 seeds (42,123,456), 1000 epochs
#   METHOD_SEEDS="42" METHOD_MAX_EPOCHS=300 bash scripts/method.sh  # Custom settings

set -e
cd /home/user

echo "Running Direct Optimization (DO) model on CaptainCook4D..."

SEEDS="${METHOD_SEEDS:-42,123,456}"
SEED_LIST=($(echo "$SEEDS" | tr ',' '\n'))
MAX_EPOCHS="${METHOD_MAX_EPOCHS:-1000}"
EARLY_STOPPING_PATIENCE="${METHOD_PATIENCE:-25}"
device='cuda'

echo "DO settings: seeds=$SEEDS, epochs=$MAX_EPOCHS, patience=$EARLY_STOPPING_PATIENCE"

mkdir -p checkpoints/do_captaincook4d

python3 << PYEOF
import sys, os, time, gc
sys.path.insert(0, '/home/user')
import numpy as np
import torch
import networkx as nx
from data.loader import load_data
from method.tgml import DirectOptimization, delete_redundant_edges, delete_cycles

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
output_dir = 'checkpoints/do_captaincook4d'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seeds = [int(s) for s in "$SEEDS".split(',')]
max_epochs = $MAX_EPOCHS
patience = $EARLY_STOPPING_PATIENCE
print(f"Device: {device}, seeds: {seeds}, max_epochs: {max_epochs}, patience: {patience}")

def post_process_for_eval(pred_core, n_gt, threshold_val):
    """Post-process predicted adjacency matrix matching official evaluation."""
    pred_binary = (pred_core > threshold_val).astype(float)
    GP = nx.DiGraph(pred_binary)
    id_start = 0
    id_end = n_gt - 1

    # Add boundary edges
    in_degree_zeros = [n for n in GP.nodes if GP.in_degree(n) == 0 and n != id_end and n != id_start]
    for node in in_degree_zeros:
        GP.add_edge(id_end, node)
    out_degree_zeros = [n for n in GP.nodes if GP.out_degree(n) == 0 and n != id_end and n != id_start]
    for node in out_degree_zeros:
        GP.add_edge(node, id_start)

    delete_redundant_edges(GP)
    delete_cycles(GP)

    pred_adj = np.zeros((n_gt, n_gt), dtype=np.float32)
    for u, v in GP.edges():
        pred_adj[u, v] = 1.0
    return pred_adj


def compute_f1(pred_adj, gt_adj):
    """Compute edge-level F1."""
    gt_flat = gt_adj.astype(np.int32).flatten()
    pred_flat = pred_adj.astype(np.int32).flatten()
    tp = int(np.sum((gt_flat == 1) & (pred_flat == 1)))
    fp = int(np.sum((gt_flat == 0) & (pred_flat == 1)))
    fn = int(np.sum((gt_flat == 1) & (pred_flat == 0)))
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


for sd in data:
    safe = sd.scenario_name.lower().replace(" ", "_")
    npz_path = os.path.join(output_dir, f"{safe}_predictions.npz")

    # Load existing seeds
    existing_cores = []
    existing_seeds = []
    if os.path.exists(npz_path):
        saved = np.load(npz_path)
        existing_cores = list(saved['pred_cores'])
        existing_seeds = list(saved['seeds'])

    seeds_to_run = [s for s in seeds if s not in existing_seeds]
    if not seeds_to_run:
        print(f"  {sd.scenario_name}: SKIP (has {len(existing_seeds)} seeds)")
        continue

    print(f"  {sd.scenario_name}: running seeds {seeds_to_run}...")
    for seed in seeds_to_run:
        np.random.seed(seed)
        torch.manual_seed(seed)
        n_gt = sd.gt_node_count
        n_matrix = n_gt + 2
        threshold_val = 1.0 / n_gt

        max_len = max(len(seq) for seq in sd.sequences)
        y = np.zeros((len(sd.sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(sd.sequences):
            y[i, :len(seq)] = seq
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)

        model = DirectOptimization(n_matrix, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        best_f1 = -1.0
        best_adj = None
        no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = model(y_tensor, beta=0.005)
            loss.backward()
            optimizer.step()

            # Evaluate F1 at each epoch (matching official approach)
            model.eval()
            with torch.no_grad():
                Z = model.get_adjacency_matrix().cpu().numpy()
            pred_core = Z[1:-1, 1:-1]
            pred_adj = post_process_for_eval(pred_core, n_gt, threshold_val)
            _, _, f1 = compute_f1(pred_adj, sd.gt_adj_matrix)

            if f1 > best_f1:
                best_f1 = f1
                best_adj = pred_core.copy()
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 100 == 0:
                print(f"    seed={seed} epoch={epoch+1}: loss={loss.item():.4f}, f1={f1:.2f}, best_f1={best_f1:.2f}")

            if no_improve >= patience:
                print(f"    seed={seed}: early stop at epoch {epoch+1}, best_f1={best_f1:.2f}")
                break

        existing_cores.append(best_adj)
        existing_seeds.append(seed)
        print(f"    seed={seed}: best_f1={best_f1:.2f}")
        del model, optimizer, y_tensor, Z
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    all_cores = np.stack(existing_cores, axis=0)
    np.savez_compressed(npz_path,
        pred_cores=all_cores,
        seeds=np.array(existing_seeds, dtype=np.int64),
    )
    print(f"  {sd.scenario_name}: saved {len(existing_seeds)} seeds")

print("=== Training done ===")
PYEOF

echo "Evaluating..."
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d/
echo "Done."
