#!/bin/bash
# CPU training on remaining 18 scenarios (6 already done).
# Batch-based: train 6, save, train 6, save, train 6, save.
# This avoids losing progress if the job is killed mid-way.

set -e
cd /home/user

echo "=== $(date) === Starting DO training on CaptainCook4D (CPU) ==="

python3 -c "import torch, numpy, networkx; print('Deps OK:', torch.__version__)"
mkdir -p checkpoints/do_captaincook4d

# === BATCH 1: scenarios 6-11 ===
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
# Scenarios 6-11: Cucumberraita, Dressedupmeatballs, Herbomeletwithfriedtomatoes, Microwaveeggsandwich, Microwavefrenchtoast, Microwavemugpizza
batch = data[6:12]
print(f"Batch 1: {[sd.scenario_name for sd in batch]}")
t0 = time.time()
for sd in batch:
    t1 = time.time()
    name, shape = train_one(sd, beta=0.005, lr=0.1, max_epochs=300, seed=42, output_dir=output_dir)
    print(f"  {name}: {shape}, {time.time()-t1:.1f}s")
print(f"Batch 1 done in {time.time()-t0:.1f}s")
PYEOF

# === BATCH 2: scenarios 12-17 ===
python3 << 'PYEOF'
import sys, os, time
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
batch = data[12:18]
print(f"Batch 2: {[sd.scenario_name for sd in batch]}")
t0 = time.time()
for sd in batch:
    t1 = time.time()
    name, shape = train_one(sd, beta=0.005, lr=0.1, max_epochs=300, seed=42, output_dir=output_dir)
    print(f"  {name}: {shape}, {time.time()-t1:.1f}s")
print(f"Batch 2 done in {time.time()-t0:.1f}s")
PYEOF

# === BATCH 3: scenarios 18-23 ===
python3 << 'PYEOF'
import sys, os, time
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
batch = data[18:24]
print(f"Batch 3: {[sd.scenario_name for sd in batch]}")
t0 = time.time()
for sd in batch:
    t1 = time.time()
    name, shape = train_one(sd, beta=0.005, lr=0.1, max_epochs=300, seed=42, output_dir=output_dir)
    print(f"  {name}: {shape}, {time.time()-t1:.1f}s")
print(f"Batch 3 done in {time.time()-t0:.1f}s")
PYEOF

echo "=== $(date) === Training complete ==="

# Evaluate
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user')
from eval.evaluator import evaluate_all_scenarios, write_scores_json
from data.loader import load_data

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
print(f'Loaded {len(data)} scenarios')

results = evaluate_all_scenarios(data, method='count_based')
print(f'Count-Based: P={results["aggregate"]["mean_precision"]:.1f}, R={results["aggregate"]["mean_recall"]:.1f}, F1={results["aggregate"]["mean_f1"]:.1f}')
write_scores_json(results, 'captaincook4d_count')

results_do = evaluate_all_scenarios(data, do_predictions_dir='checkpoints/do_captaincook4d', method='do')
print(f'DO: P={results_do["aggregate"]["mean_precision"]:.1f}, R={results_do["aggregate"]["mean_recall"]:.1f}, F1={results_do["aggregate"]["mean_f1"]:.1f}, std={results_do["aggregate"]["std_f1"]:.1f}')
write_scores_json(results_do, 'captaincook4d_do')
print('Scores written to scoring/scores.json')
PYEOF

echo "=== $(date) === Validation ==="
python3 validate.py --compare
echo "Done."
