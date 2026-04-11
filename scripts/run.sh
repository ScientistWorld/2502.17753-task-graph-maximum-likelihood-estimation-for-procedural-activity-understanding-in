#!/bin/bash
# Train remaining 4 scenarios (Spicytunaavocadowraps, Tomatochutney, Tomatomozzarellasalad, Zoodles)
set -e
cd /home/user
echo "=== $(date) === DO training final 4 ==="

python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); torch.set_num_threads(1)"
mkdir -p checkpoints/do_captaincook4d

python3 << 'PYEOF'
import sys, os, time, gc
sys.path.insert(0, '/home/user')
import numpy as np
import torch
torch.set_num_threads(1)

from data.loader import load_data
from method.tgml import DirectOptimization

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
output_dir = 'checkpoints/do_captaincook4d'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Remaining 4: indices 20, 21, 22, 23
remaining = [d for d in data if d.scenario_name.lower().replace(" ", "_") not in [
    'blenderbananapancakes', 'breakfastburritos', 'broccolistirfry', 'buttercorncup',
    'capresebruschetta', 'cheesepimiento', 'coffee', 'cucumberraita', 'dressedupmeatballs',
    'herbomeletwithfriedtomatoes', 'microwaveeggsandwich', 'microwavefrenchtoast',
    'microwavemugpizza', 'mugcake', 'panfriedtofu', 'pinwheels', 'ramen',
    'sautedmushrooms', 'scrambledeggs', 'spicedhotchocolate',
]]
print(f"Remaining {len(remaining)}: {[sd.scenario_name for sd in remaining]}")

t0 = time.time()
for sd in remaining:
    safe = sd.scenario_name.lower().replace(" ", "_")
    npz_path = os.path.join(output_dir, f"{safe}_predictions.npz")
    if os.path.exists(npz_path):
        print(f"  {sd.scenario_name}: SKIP")
        continue

    t1 = time.time()
    np.random.seed(42)
    torch.manual_seed(42)
    n_gt = sd.gt_node_count
    n_matrix = n_gt + 2
    max_len = max(len(seq) for seq in sd.sequences)
    y = np.zeros((len(sd.sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sd.sequences):
        y[i, :len(seq)] = seq

    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    model = DirectOptimization(n_matrix, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(300):
        optimizer.zero_grad()
        loss = model(y_tensor, beta=0.005)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Z = model.get_adjacency_matrix().cpu().numpy()

    pred_core = Z[1:-1, 1:-1]
    np.savez_compressed(npz_path,
        pred_cores=np.array([pred_core], dtype=np.float32),
        seeds=np.array([42], dtype=np.int64),
    )
    print(f"  {sd.scenario_name}: {time.time()-t1:.1f}s")
    del model, optimizer, y_tensor, Z, pred_core, y
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

print(f"=== Training done: {time.time()-t0:.1f}s ===")
PYEOF

# Evaluate
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user')
from eval.evaluator import evaluate_all_scenarios, write_scores_json
from data.loader import load_data

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
results = evaluate_all_scenarios(data, method='count_based')
print(f'Count: P={results["aggregate"]["mean_precision"]:.1f}, R={results["aggregate"]["mean_recall"]:.1f}, F1={results["aggregate"]["mean_f1"]:.1f}')
write_scores_json(results, 'captaincook4d_count')

results_do = evaluate_all_scenarios(data, do_predictions_dir='checkpoints/do_captaincook4d', method='do')
print(f'DO: P={results_do["aggregate"]["mean_precision"]:.1f}, R={results_do["aggregate"]["mean_recall"]:.1f}, F1={results_do["aggregate"]["mean_f1"]:.1f}, std={results_do["aggregate"]["std_f1"]:.1f}')
write_scores_json(results_do, 'captaincook4d_do')
PYEOF

python3 validate.py --compare
echo "=== Done ==="
