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

echo "Running Direct Optimization (DO) model on CaptainCook4D..."

SEEDS="${METHOD_SEEDS:-42,1337,2024,2025,2026}"
SEED_LIST=($(echo "$SEEDS" | tr ',' '\n'))
MAX_EPOCHS="${METHOD_MAX_EPOCHS:-1000}"
DEVICE="${METHOD_DEVICE:-cpu}"
OUTPUT_DIR="${METHOD_OUTPUT_DIR:-checkpoints/do_captaincook4d_fixedtgml}"
SCENARIO_FILTER="${METHOD_SCENARIOS:-}"
MAX_SCENARIOS="${METHOD_MAX_SCENARIOS:-0}"

echo "DO settings: seeds=$SEEDS, epochs=$MAX_EPOCHS, device=$DEVICE, output=$OUTPUT_DIR, scenarios=${SCENARIO_FILTER:-ALL}, max_scenarios=$MAX_SCENARIOS"

mkdir -p "$OUTPUT_DIR"

python3 << PYEOF
import sys, os, time, gc
sys.path.insert(0, '/home/user')
import numpy as np
import torch
from data.loader import load_data
from method.tgml import DirectOptimization

data = load_data(repo_data_path='/home/user/method/repo/data/captaincook4d')
output_dir = "$OUTPUT_DIR"
device = "$DEVICE"
if device != 'cpu' and not torch.cuda.is_available():
    device = 'cpu'
seeds = [int(s) for s in "$SEEDS".split(',')]
max_epochs = $MAX_EPOCHS
scenario_filter = [s.strip().lower() for s in "$SCENARIO_FILTER".split(',') if s.strip()]
max_scenarios = int("$MAX_SCENARIOS")
if scenario_filter:
    data = [
        sd for sd in data
        if sd.scenario_name.lower().replace(" ", "_") in scenario_filter
        or sd.scenario_name.lower() in scenario_filter
    ]
print(f"Device: {device}, seeds: {seeds}, max_epochs: {max_epochs}, scenarios={len(data)}")

def safe_name(sd):
    return sd.scenario_name.lower().replace(" ", "_")

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

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = model(y_tensor, beta=0.005)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"    seed={seed} epoch={epoch+1}: loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            Z = model.get_adjacency_matrix().cpu().numpy()
        pred_core = Z[1:-1, 1:-1].copy()

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

echo "Evaluating..."
bash scripts/evaluate.sh captaincook4d_do "$OUTPUT_DIR/"
echo "Done."
