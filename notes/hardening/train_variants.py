#!/usr/bin/env python3
"""Hardening audit: honest variants of the reference, trained from the final code.

Run after scripts/method.sh has written the reference to checkpoints/do.
Writes notes/hardening/runs/<variant>/<procedure>_predictions.npz (5 trials each):

  do_rerun_seedset_{b,c,d}  reference protocol, three other 5-seed sets (rerun noise)
  do_best_sa_checkpoint     reference seeds, state at the best-SA epoch instead of
                            the halting epoch (the redteam's "better" variant)
  do_one_step               reference seeds, state after one Adam update (weaker)
  do_seed_ensemble          mean Z of the 5 reference seeds, repeated 5x
  do_seed_ensemble20        mean Z of all 20 seeds, repeated 5x
  always_before_hasse       transitive reduction of the "always observed before"
                            relation of the demonstrations (no learning), 5x
  do_beta_0p05 / do_beta_0p5  reference protocol and seeds with beta 0.05 / 0.5
                            (the paper's beta ablation, Fig. 7)

Also re-derives the reference seeds and checks they reproduce checkpoints/do
bit-for-bit (determinism of the reference).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/user")
sys.path.insert(0, str(ROOT))

from data.loader import load_data  # noqa: E402
from method.train import DEFAULT_DATA, DEFAULT_SEEDS, procedure_key, train_do  # noqa: E402

RUNS = ROOT / "notes" / "hardening" / "runs"
RESULTS = ROOT / "notes" / "hardening" / "results"
REFERENCE_DIR = ROOT / "checkpoints" / "do"
BETA_VARIANTS = {"do_beta_0p05": 0.05, "do_beta_0p5": 0.5}
SEED_SETS = {
    "do_rerun_seedset_b": [1, 2, 3, 4, 5],
    "do_rerun_seedset_c": [6, 7, 8, 9, 10],
    "do_rerun_seedset_d": [11, 12, 13, 14, 15],
}


def save(variant: str, key: str, cores, seeds) -> None:
    out = RUNS / variant
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{key}_predictions.npz",
                        pred_cores=np.stack(cores).astype(np.float32),
                        seeds=np.array(seeds, dtype=np.int64))


def always_before_hasse(scenario_data) -> np.ndarray:
    """Edge d->p (p precondition of d) for every pair where p precedes d in every
    demonstration containing both, transitively reduced."""
    n = scenario_data.gt_node_count
    sequences = [[step - 1 for step in seq[1:-1]] for seq in scenario_data.sequences]
    positions = [{node: i for i, node in enumerate(seq)} for seq in sequences]
    rel = np.zeros((n, n), dtype=bool)
    for d in range(n):
        for p in range(n):
            if d == p:
                continue
            both = [pos for pos in positions if d in pos and p in pos]
            if both and all(pos[p] < pos[d] for pos in both):
                rel[d, p] = True
    reach = rel.copy()
    for k in range(n):
        reach |= reach[:, [k]] & reach[[k], :]
    reduced = rel.copy()
    for a in range(n):
        for b in range(n):
            if rel[a, b] and any(c != a and c != b and reach[a, c] and reach[c, b] for c in range(n)):
                reduced[a, b] = False
    return reduced.astype(np.float32)


def main() -> None:
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_data(repo_data_path=DEFAULT_DATA)
    metadata = {}
    determinism = {}
    started = time.time()
    for sd in data:
        key = procedure_key(sd)
        ref_runs = [train_do(sd, seed=seed) for seed in DEFAULT_SEEDS]
        saved = np.load(REFERENCE_DIR / f"{key}_predictions.npz")["pred_cores"]
        determinism[key] = bool(all(np.array_equal(saved[i], run["pred_core"]) for i, run in enumerate(ref_runs)))

        extra = {name: [train_do(sd, seed=seed) for seed in seeds] for name, seeds in SEED_SETS.items()}
        for name, beta in BETA_VARIANTS.items():
            save(name, key, [train_do(sd, seed=seed, beta=beta)["pred_core"] for seed in DEFAULT_SEEDS],
                 DEFAULT_SEEDS)

        save("do_best_sa_checkpoint", key, [r["best_core"] for r in ref_runs], DEFAULT_SEEDS)
        save("do_one_step", key, [r["first_core"] for r in ref_runs], DEFAULT_SEEDS)
        for name, runs in extra.items():
            save(name, key, [r["pred_core"] for r in runs], SEED_SETS[name])
        ensemble5 = np.mean([r["pred_core"] for r in ref_runs], axis=0)
        save("do_seed_ensemble", key, [ensemble5] * 5, DEFAULT_SEEDS)
        all_cores = [r["pred_core"] for r in ref_runs] + [r["pred_core"] for runs in extra.values() for r in runs]
        save("do_seed_ensemble20", key, [np.mean(all_cores, axis=0)] * 5, [0] * 5)
        save("always_before_hasse", key, [always_before_hasse(sd)] * 5, [0] * 5)

        metadata[key] = {
            "reference": [{k: r[k] for k in ("seed", "stop_epoch", "best_epoch", "best_sequence_accuracy",
                                             "final_sequence_accuracy")} for r in ref_runs],
            **{name: [{k: r[k] for k in ("seed", "stop_epoch", "best_epoch", "final_sequence_accuracy")}
                      for r in runs] for name, runs in extra.items()},
        }
        print(f"{sd.scenario_name}: reference stop epochs {[r['stop_epoch'] for r in ref_runs]}, "
              f"deterministic={determinism[key]}, elapsed {time.time() - started:.0f}s", flush=True)

    with open(RESULTS / "training_metadata.json", "w") as f:
        json.dump({"determinism_vs_checkpoints_do": determinism, "runs": metadata}, f, indent=2)
    if not all(determinism.values()):
        print("WARNING: reference re-derivation differs from checkpoints/do for",
              [k for k, v in determinism.items() if not v])


if __name__ == "__main__":
    main()
