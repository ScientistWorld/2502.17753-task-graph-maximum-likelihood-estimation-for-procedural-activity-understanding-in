#!/bin/bash
# GPU training on all 24 scenarios sequentially.
# Use CUDA with sequential processing to avoid multi-GPU overhead.

set -e
cd /home/user
echo "=== $(date) === DO training on GPU ==="

python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
mkdir -p checkpoints/do_captaincook4d

python3 << 'PYEOF'
import sys, os, time, gc
sys.path.insert(0, '/home/user')
import numpy as np
import torch
torch.set_num_threads(1)
from data.loader import load_data
from method.tgml import DirectOptimization

def train_one(scenario_data, beta, lr, max_epochs, seed, output_dir):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gt = scenario_data.gt_node_count
    n_matrix = n_gt + 2
    max_len = max(len(seq) for seq in scenario_data.sequences)
    y = np.zeros((len(scenario_data.sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(scenario_data.sequences):
        y[i, :len(seq)] = seq
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    model = DirectOptimization(n_matrix, device=device).to(device)
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
    safe = scenario_data.scenario_name.lower().replace(" ", "_")
    np.savez_compressed(
        os.path.join(output_dir, f"{safe}_predictions.npz"),
        pred_cores=np.array([pred_core], dtype=np.float32),
        seeds=np.array([seed], dtype=np.int64),
    )
    del model, optimizer, y_tensor, Z, pred_core
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    return scenario_data.scenario_name

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
output_dir = 'checkpoints/do_captaincook4d'
os.makedirs(output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loaded {len(data)} scenarios, device={device}")

t0 = time.time()
for idx, sd in enumerate(data):
    t1 = time.time()
    name = train_one(sd, beta=0.005, lr=0.1, max_epochs=300, seed=42, output_dir=output_dir)
    elapsed = time.time() - t1
    total = time.time() - t0
    print(f"  [{idx+1}/24] {name}: {elapsed:.1f}s (total: {total:.1f}s)")
print(f"=== All 24 trained in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}min) ===")
PYEOF

echo "=== $(date) === Evaluating ==="
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user')
from eval.evaluator import evaluate_all_scenarios, write_scores_json
from data.loader import load_data

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
results = evaluate_all_scenarios(data, method='count_based')
print(f'Count-Based: P={results["aggregate"]["mean_precision"]:.1f}, R={results["aggregate"]["mean_recall"]:.1f}, F1={results["aggregate"]["mean_f1"]:.1f}')
write_scores_json(results, 'captaincook4d_count')

results_do = evaluate_all_scenarios(data, do_predictions_dir='checkpoints/do_captaincook4d', method='do')
print(f'DO: P={results_do["aggregate"]["mean_precision"]:.1f}, R={results_do["aggregate"]["mean_recall"]:.1f}, F1={results_do["aggregate"]["mean_f1"]:.1f}, std={results_do["aggregate"]["std_f1"]:.1f}')
write_scores_json(results_do, 'captaincook4d_do')
PYEOF

echo "=== $(date) === Done ==="
python3 validate.py --compare
