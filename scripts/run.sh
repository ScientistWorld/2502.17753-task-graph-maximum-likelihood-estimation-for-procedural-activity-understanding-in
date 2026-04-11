#!/bin/bash
# Train seeds 123 and 456 for all 24 CaptainCook4D scenarios, then evaluate.
# Seed 42 already done. This adds seeds 123 and 456, then averages all 3.
set -e
cd /home/user
echo "=== $(date) === DO multi-seed training (seeds 123, 456) ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

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
extra_seeds = [123, 456]

for sd in data:
    safe = sd.scenario_name.lower().replace(" ", "_")
    npz_path = os.path.join(output_dir, f"{safe}_predictions.npz")

    # Load existing predictions
    if os.path.exists(npz_path):
        saved = np.load(npz_path)
        existing_seeds = set(saved['seeds'].tolist())
    else:
        existing_seeds = set()

    seeds_to_run = [s for s in extra_seeds if s not in existing_seeds]
    if not seeds_to_run:
        print(f"  {sd.scenario_name}: SKIP (seeds {existing_seeds} done)")
        continue

    print(f"  {sd.scenario_name}: training seeds {seeds_to_run}...")
    pred_cores = []

    # Load existing predictions
    if os.path.exists(npz_path):
        saved = np.load(npz_path)
        pred_cores.extend(saved['pred_cores'].tolist())
        all_seeds = saved['seeds'].tolist()
    else:
        all_seeds = []

    for seed in seeds_to_run:
        t1 = time.time()
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
        for epoch in range(300):
            optimizer.zero_grad()
            loss = model(y_tensor, beta=0.005)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            Z = model.get_adjacency_matrix().cpu().numpy()
        pred_core = Z[1:-1, 1:-1]
        pred_cores.append(pred_core)
        all_seeds.append(seed)
        del model, optimizer, y_tensor, Z, y
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        print(f"    seed {seed}: {time.time()-t1:.1f}s")

    all_cores = np.stack(pred_cores, axis=0).astype(np.float32)
    np.savez_compressed(npz_path,
        pred_cores=all_cores,
        seeds=np.array(all_seeds, dtype=np.int64),
    )
    print(f"  {sd.scenario_name}: saved {len(all_seeds)} seeds total")

print("=== Training done ===")
PYEOF

# Evaluate
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/home/user')
from eval.evaluator import evaluate_all_scenarios, write_scores_json
from data.loader import load_data

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')

# Count-based
results = evaluate_all_scenarios(data, method='count_based')
print(f'Count: P={results["aggregate"]["mean_precision"]:.1f}, R={results["aggregate"]["mean_recall"]:.1f}, F1={results["aggregate"]["mean_f1"]:.1f}')
write_scores_json(results, 'captaincook4d_count')

# DO with 3-seed averaging
results_do = evaluate_all_scenarios(data, do_predictions_dir='checkpoints/do_captaincook4d', method='do')
print(f'DO: P={results_do["aggregate"]["mean_precision"]:.1f}, R={results_do["aggregate"]["mean_recall"]:.1f}, F1={results_do["aggregate"]["mean_f1"]:.1f}, std={results_do["aggregate"]["std_f1"]:.1f}')
write_scores_json(results_do, 'captaincook4d_do')
PYEOF
python3 validate.py --compare
echo "=== Done ==="
