#!/bin/bash
# Job script: CPU training on remaining 20 scenarios (first 4 already have predictions).
# Trains sequentially to avoid memory/CUDA issues.

set -e
cd /home/user

echo "=== $(date) === Starting DO training on CaptainCook4D (CPU) ==="

# Verify dependencies
python3 -c "import torch, numpy, networkx; print('Deps OK:', torch.__version__)"

mkdir -p checkpoints/do_captaincook4d

# Train remaining scenarios (skip those with existing predictions)
python3 << 'PYEOF'
import sys, os, time, json
sys.path.insert(0, '/home/user')
import numpy as np
import torch
from data.loader import load_data
from method.tgml import DirectOptimization

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

# Skip first 4 scenarios (already have predictions from previous run)
remaining = data[4:]
print(f"Training remaining {len(remaining)} scenarios...")

t0 = time.time()
for idx, sd in enumerate(remaining):
    t1 = time.time()
    name, shape = train_one(sd, beta=0.005, lr=0.1, max_epochs=300, seed=42, output_dir=output_dir)
    elapsed = time.time() - t1
    print(f"  [{idx+1}/20] {name}: {shape}, {elapsed:.1f}s")
total_elapsed = time.time() - t0
print(f"=== Remaining 20 trained in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min) ===")

# Verify all 24 predictions exist
npz_files = [f for f in os.listdir(output_dir) if f.endswith('_predictions.npz')]
print(f"Total prediction files: {len(npz_files)}/24")
PYEOF

echo "Training complete. Evaluating..."

# Evaluate
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user')
from eval.evaluator import evaluate_all_scenarios, write_scores_json
from data.loader import load_data

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
print(f'Loaded {len(data)} scenarios')

# Count-based
results = evaluate_all_scenarios(data, method='count_based')
print(f'Count-Based: P={results["aggregate"]["mean_precision"]:.1f}, R={results["aggregate"]["mean_recall"]:.1f}, F1={results["aggregate"]["mean_f1"]:.1f}')
write_scores_json(results, 'captaincook4d_count')

# DO
results_do = evaluate_all_scenarios(data, do_predictions_dir='checkpoints/do_captaincook4d', method='do')
print(f'DO: P={results_do["aggregate"]["mean_precision"]:.1f}, R={results_do["aggregate"]["mean_recall"]:.1f}, F1={results_do["aggregate"]["mean_f1"]:.1f}, std={results_do["aggregate"]["std_f1"]:.1f}')
write_scores_json(results_do, 'captaincook4d_do')
print('Scores written to scoring/scores.json')
PYEOF

echo "=== $(date) === Validation ==="
python3 validate.py --compare
echo "Done."
