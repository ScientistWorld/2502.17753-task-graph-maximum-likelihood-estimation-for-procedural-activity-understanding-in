#!/bin/bash
# Run the paper's proposed DO method on CaptainCook4D.
# This keeps training weakly supervised: optimize TGML on action sequences,
# then evaluate only after training completes.
#
# Usage:
#   bash scripts/method.sh
#   METHOD_SEEDS="42" METHOD_MAX_EPOCHS=300 bash scripts/method.sh

set -e
cd /home/user
source /home/user/scripts/common_env.sh

echo "Running Direct Optimization (DO) model on CaptainCook4D..."

SEEDS="${METHOD_SEEDS:-42,1337,2024,2025,2026}"
SEED_LIST=($(echo "$SEEDS" | tr ',' '\n'))
MAX_EPOCHS="${METHOD_MAX_EPOCHS:-1000}"
DEVICE="${METHOD_DEVICE:-cpu}"
OUTPUT_DIR="${METHOD_OUTPUT_DIR:-checkpoints/do_captaincook4d_fixedtgml}"
SCENARIO_FILTER="${METHOD_SCENARIOS:-}"
MAX_SCENARIOS="${METHOD_MAX_SCENARIOS:-0}"
SELECTION="${METHOD_SELECTION:-final}"
EARLY_STOP_PATIENCE="${METHOD_EARLY_STOP_PATIENCE:-25}"

echo "DO settings: seeds=$SEEDS, epochs=$MAX_EPOCHS, device=$DEVICE, output=$OUTPUT_DIR, scenarios=${SCENARIO_FILTER:-ALL}, max_scenarios=$MAX_SCENARIOS, selection=$SELECTION"

mkdir -p "$OUTPUT_DIR"

python3 << PYEOF
import sys, os, time, gc, copy
sys.path.insert(0, '/home/user')
import numpy as np
import networkx as nx
import torch
from data.loader import load_data
from method.tgml import DirectOptimization

data = load_data(repo_data_path='/home/user/data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d')
output_dir = "$OUTPUT_DIR"
device = "$DEVICE"
if device != 'cpu' and not torch.cuda.is_available():
    device = 'cpu'
seeds = [int(s) for s in "$SEEDS".split(',')]
max_epochs = $MAX_EPOCHS
scenario_filter = [s.strip().lower() for s in "$SCENARIO_FILTER".split(',') if s.strip()]
max_scenarios = int("$MAX_SCENARIOS")
selection = "$SELECTION"
early_stop_patience = int("$EARLY_STOP_PATIENCE")
if scenario_filter:
    data = [
        sd for sd in data
        if sd.scenario_name.lower().replace(" ", "_") in scenario_filter
        or sd.scenario_name.lower() in scenario_filter
    ]
print(f"Device: {device}, seeds: {seeds}, max_epochs: {max_epochs}, scenarios={len(data)}")

def safe_name(sd):
    return sd.scenario_name.lower().replace(" ", "_")

def postprocess_for_sequence_accuracy(pred_core, n_gt):
    threshold = 1.0 / n_gt
    pred_binary = np.where(pred_core < threshold, 0, 1).astype(float)
    graph = nx.DiGraph(pred_binary)
    id_start = 0
    id_end = n_gt - 1
    for node in [
        node for node in graph.nodes
        if graph.in_degree(node) == 0 and node != id_end and node != id_start
    ]:
        graph.add_edge(id_end, node)
    for node in [
        node for node in graph.nodes
        if graph.out_degree(node) == 0 and node != id_end and node != id_start
    ]:
        graph.add_edge(node, id_start)

    graph_copy = copy.deepcopy(graph)
    for current_node in graph.nodes:
        for successor in list(graph_copy.successors(current_node)):
            if graph.has_edge(current_node, successor):
                graph.remove_edge(current_node, successor)
                if not nx.has_path(graph, current_node, successor):
                    graph.add_edge(current_node, successor)
    return graph

def sequence_accuracy(sequences, predecessors):
    accuracy = 0.0
    for sequence in sequences:
        correct = 0.0
        for index, node in enumerate(sequence):
            previous = set(sequence[:index])
            predicted = set(predecessors.get(node, []))
            if len(previous) == 0 and len(predicted) == 0:
                correct += 1.0
            elif len(previous) > 0 and len(predicted) > 0:
                correct += len(previous.intersection(predicted)) / len(predicted)
        accuracy += correct / len(sequence)
    return accuracy / len(sequences) if sequences else 0.0

pending = []
complete = []
for sd in data:
    npz_path = os.path.join(output_dir, f"{safe_name(sd)}_predictions.npz")
    existing_seeds = []
    if os.path.exists(npz_path):
        saved = np.load(npz_path)
        existing_seeds = list(saved['seeds'])
    missing = [s for s in seeds if s not in existing_seeds]
    if missing:
        workload = sum(len(seq) for seq in sd.sequences) * max(sd.gt_node_count, 1)
        pending.append((len(missing), workload, sd))
    else:
        complete.append(sd.scenario_name)

pending.sort(key=lambda x: (x[0], x[1], x[2].gt_node_count, x[2].scenario_name))
if max_scenarios > 0:
    pending = pending[:max_scenarios]
if complete:
    print(f"Already complete: {len(complete)} scenarios")
if pending:
    print("Pending order:", [sd.scenario_name for _, _, sd in pending])

for _, _, sd in pending:
    safe = safe_name(sd)
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

        max_len = max(len(seq) for seq in sd.sequences)
        y = np.zeros((len(sd.sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(sd.sequences):
            y[i, :len(seq)] = seq
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)

        model = DirectOptimization(n_matrix, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train_seq_no_beta_gamma = [
            [node - 1 for node in seq if 1 <= node <= n_gt]
            for seq in sd.sequences
        ]
        best_core = None
        best_seq_acc = -float("inf")
        early_stopping_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = model(y_tensor, beta=0.005)
            loss.backward()
            optimizer.step()

            if selection == "sequence_accuracy":
                with torch.no_grad():
                    pred_core_epoch = model.get_adjacency_matrix().cpu().numpy()[1:-1, 1:-1].copy()
                graph = postprocess_for_sequence_accuracy(pred_core_epoch, n_gt)
                seq_acc = 0.0
                if nx.is_directed_acyclic_graph(graph):
                    predecessors = {node: list(graph.successors(node)) for node in graph.nodes}
                    seq_acc = sequence_accuracy(train_seq_no_beta_gamma, predecessors)
                    if seq_acc > 0.95:
                        if seq_acc == 1.0 and seq_acc == best_seq_acc:
                            early_stopping_counter += 1
                        elif seq_acc >= best_seq_acc:
                            best_seq_acc = seq_acc
                            best_core = pred_core_epoch
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                if early_stopping_counter >= early_stop_patience:
                    print(f"    seed={seed}: early stop at epoch={epoch+1}, best_seq_acc={best_seq_acc:.4f}")
                    break

            if (epoch + 1) % 100 == 0:
                print(f"    seed={seed} epoch={epoch+1}: loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            Z = model.get_adjacency_matrix().cpu().numpy()
        pred_core = Z[1:-1, 1:-1].copy()
        if selection == "sequence_accuracy" and best_core is not None:
            pred_core = best_core.copy()
            print(f"    seed={seed}: selected best sequence-accuracy adjacency ({best_seq_acc:.4f})")

        existing_cores.append(pred_core)
        existing_seeds.append(seed)

        order = np.argsort(np.array(existing_seeds, dtype=np.int64))
        ordered_cores = np.stack([existing_cores[i] for i in order], axis=0)
        ordered_seeds = np.array([existing_seeds[i] for i in order], dtype=np.int64)
        np.savez_compressed(npz_path,
            pred_cores=ordered_cores,
            seeds=ordered_seeds,
        )
        print(f"    seed={seed}: saved final adjacency")
        del model, optimizer, y_tensor, Z
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    print(f"  {sd.scenario_name}: saved {len(existing_seeds)} seeds")

print("=== Training done ===")
PYEOF

TOTAL_SCENARIOS="$(python3 - <<'PY'
import sys
import contextlib
import io
sys.path.insert(0, '/home/user')
from data.loader import load_data
with contextlib.redirect_stdout(io.StringIO()):
    data = load_data(repo_data_path='/home/user/data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d')
print(len(data))
PY
)"

COMPLETED_SCENARIOS="$(python3 - <<PY
import os
import numpy as np

pred_dir = "$OUTPUT_DIR"
expected = sorted(int(s) for s in "$SEEDS".split(','))
complete = 0
if os.path.isdir(pred_dir):
    for fn in os.listdir(pred_dir):
        if not fn.endswith("_predictions.npz"):
            continue
        data = np.load(os.path.join(pred_dir, fn))
        seeds = sorted(int(x) for x in data["seeds"])
        if seeds == expected:
            complete += 1
print(complete)
PY
)"

if [ "$COMPLETED_SCENARIOS" -eq "$TOTAL_SCENARIOS" ]; then
  echo "Evaluating full CaptainCook4D DO result..."
  bash scripts/evaluate.sh captaincook4d_do "$OUTPUT_DIR/"
else
  echo "Skipping full DO evaluation; only $COMPLETED_SCENARIOS/$TOTAL_SCENARIOS scenarios have all requested seeds."
fi
echo "Done."
