#!/bin/bash
# Job script: sequential DO training across all 24 CaptainCook4D scenarios.
# Each scenario trains independently and saves predictions immediately,
# so even partial progress is preserved.

set -e
cd /home/user

echo "=== $(date) === Starting DO training on CaptainCook4D (24 scenarios, 300 epochs, sequential) ==="

# Verify dependencies
python3 -c "import torch, numpy, networkx; print('Deps OK:', torch.__version__)"

# Clean slate for fresh scores
rm -f scoring/scores.json
mkdir -p checkpoints/do_captaincook4d

# Train each scenario sequentially.
# We avoid multiprocessing to prevent fork overhead on compute nodes.
python3 << 'PYEOF'
import sys, os, time, json
sys.path.insert(0, '/home/user')
import numpy as np
import torch

from data.loader import load_data
from method.tgml import DirectOptimization

def post_process(pred_core, n_gt, threshold):
    import networkx as nx
    from method.tgml import delete_redundant_edges, delete_cycles
    pred_binary = (pred_core > threshold).astype(float)
    G = nx.DiGraph(pred_binary.T)
    id_start, id_end = 0, n_gt - 1
    # Add boundary edges
    for node in [n for n in G.nodes if G.in_degree(n) == 0 and n not in (id_start, id_end)]:
        G.add_edge(id_end, node)
    for node in [n for n in G.nodes if G.out_degree(n) == 0 and n not in (id_start, id_end)]:
        G.add_edge(node, id_start)
    delete_redundant_edges(G)
    delete_cycles(G)
    return G

def train_one(scenario_data, beta, lr, max_epochs, seed, output_dir):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gt = scenario_data.gt_node_count
    n_matrix = n_gt + 2
    max_len = max(len(seq) for seq in scenario_data.sequences)
    y = np.zeros((len(scenario_data.sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(scenario_data.sequences):
        y[i, :len(seq)] = seq
    y_tensor = torch.tensor(y, dtype=torch.long, device='cpu')
    model = DirectOptimization(n_matrix, device='cpu').to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    safe = scenario_data.scenario_name.lower().replace(" ", "_")
    np.savez_compressed(
        os.path.join(output_dir, f"{safe}_predictions.npz"),
        pred_cores=np.array([pred_core], dtype=np.float32),
        seeds=np.array([seed], dtype=np.int64),
    )
    return scenario_data.scenario_name, pred_core.shape

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
output_dir = 'checkpoints/do_captaincook4d'
os.makedirs(output_dir, exist_ok=True)

print(f"Loaded {len(data)} scenarios")
total_start = time.time()
for idx, sd in enumerate(data):
    t0 = time.time()
    name, shape = train_one(sd, beta=0.005, lr=0.1, max_epochs=300, seed=42, output_dir=output_dir)
    elapsed = time.time() - t0
    print(f"  [{idx+1}/24] {name}: {shape}, {elapsed:.1f}s")

total_elapsed = time.time() - total_start
print(f"=== All 24 scenarios trained in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min) ===")

# Save overall timing
with open(os.path.join(output_dir, "overall_results.json"), "w") as f:
    json.dump({"dataset": "captaincook4d", "elapsed_seconds": total_elapsed,
              "num_scenarios": len(data), "max_epochs": 300, "seed": 42}, f, indent=2)
PYEOF

echo "Training complete. Evaluating..."
bash scripts/evaluate.sh captaincook4d_do checkpoints/do_captaincook4d/

echo "=== $(date) === Validation ==="
python3 validate.py --compare
echo "Done."
