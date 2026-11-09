#!/usr/bin/env python3
"""Hardening audit: build the shortcut submissions that are replayed through the
hidden test. Each case is a prediction directory under notes/hardening/runs/sc_*;
cases that also tamper with editable files are applied by run_hardening.sh.

  sc_visible_labels        copy whatever ground-truth edges the public files still hold
  sc_rewrite_ground_truth  all-zero cores (+ public held-out graph files rewritten to match)
  sc_partial_coverage      reference cores for one procedure per slice only
                           (+ scores_train.json registry key "captaincook4d_partial_compare")
  sc_seed_ids              reference cores, every declared seed id set to 0
  sc_single_seed           reference seed 42 only (one trial)
  sc_single_seed_x5        reference seed 42 duplicated into all five trials
  sc_forge_scores          all-zero cores (+ forged score files written beforehand)
  sc_zero / sc_ones        constant cores (valid format, no inference)
  sc_nan / sc_inf / sc_neginf / sc_empty / sc_out_of_range / sc_wrong_shape
                           impossible artifacts
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
sys.path.insert(0, str(ROOT))

from data.loader import load_data  # noqa: E402
from method.train import DEFAULT_DATA, procedure_key  # noqa: E402

RUNS = ROOT / "notes" / "hardening" / "runs"
REFERENCE_DIR = ROOT / "checkpoints" / "do"
PUBLIC_GRAPH_DIRS = [
    Path(DEFAULT_DATA) / "ground_truth_task_graphs",
    ROOT / "method" / "repo" / "data" / "captaincook4d" / "ground_truth_task_graphs",
]
PARTIAL_KEYS = {"spicedhotchocolate", "tomatomozzarellasalad"}  # smallest held-out / public procedure


def save(case: str, key: str, cores, seeds=None) -> None:
    out = RUNS / case
    out.mkdir(parents=True, exist_ok=True)
    cores = np.asarray(cores)
    if seeds is None:
        seeds = list(range(cores.shape[0]))
    np.savez_compressed(out / f"{key}_predictions.npz", pred_cores=cores, seeds=np.array(seeds, dtype=np.int64))


def public_edges(key: str):
    for graph_dir in PUBLIC_GRAPH_DIRS:
        path = graph_dir / f"{key}.json"
        if path.is_file():
            graph = json.loads(path.read_text())
            if graph.get("edges"):
                return graph["edges"]
    return []


def main() -> None:
    report = {}
    for sd in load_data(repo_data_path=DEFAULT_DATA):
        key = procedure_key(sd)
        n = sd.gt_node_count
        ref = np.load(REFERENCE_DIR / f"{key}_predictions.npz")["pred_cores"]
        zeros = np.zeros((n, n), dtype=np.float32)

        edges = public_edges(key)
        copied = np.zeros((n, n), dtype=np.float32)
        for a, b in edges:
            copied[int(b), int(a)] = 1.0
        report[key] = {"public_edges_found": len(edges)}
        save("sc_visible_labels", key, [copied] * 5)

        save("sc_rewrite_ground_truth", key, [zeros] * 5)
        save("sc_forge_scores", key, [zeros] * 5)
        save("sc_zero", key, [zeros] * 5)
        save("sc_ones", key, [np.ones((n, n), dtype=np.float32)] * 5)
        save("sc_nan", key, [np.full((n, n), np.nan, dtype=np.float32)] * 5)
        save("sc_inf", key, [np.full((n, n), np.inf, dtype=np.float32)] * 5)
        save("sc_neginf", key, [np.full((n, n), -np.inf, dtype=np.float32)] * 5)
        save("sc_empty", key, np.zeros((0, n, n), dtype=np.float32), seeds=[])
        save("sc_out_of_range", key, ref * 4.0)
        save("sc_wrong_shape", key, ref[:, :-1, :-1])
        save("sc_seed_ids", key, ref, seeds=[0] * ref.shape[0])
        save("sc_single_seed", key, ref[:1], seeds=[42])
        save("sc_single_seed_x5", key, np.repeat(ref[:1], 5, axis=0), seeds=[42] * 5)
        if key in PARTIAL_KEYS:
            save("sc_partial_coverage", key, ref)

    with open(ROOT / "notes" / "hardening" / "results" / "shortcut_build.json", "w") as f:
        json.dump(report, f, indent=2)
    print("public held-out edges found:",
          {k: v["public_edges_found"] for k, v in report.items() if v["public_edges_found"] == 0})


if __name__ == "__main__":
    main()
