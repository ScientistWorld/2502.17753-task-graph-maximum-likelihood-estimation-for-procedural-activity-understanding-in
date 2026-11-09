#!/usr/bin/env python3
"""Choose the noise level of the noisy-demonstration experiment (public split only).

Predeclared rule: candidate levels p in {0.20, 0.10, 0.05} are tried from the
hardest down; the scored level is the first one at which the paper's reference
(DO, paper protocol, 5 seeds) beats the evaluator's Count-Based baseline by at
least 5 F1 points on the 40 PUBLIC (train) procedures. The experiment should be
as hard as possible (headroom where the paper's method falls short) while the
paper's claim over its baseline still holds. No held-out graph is read.
The chosen level is installed with eval/test/build_noisy_benchmark.py finalize.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/user")
sys.path.insert(0, str(ROOT))

from data.loader import load_noisy_demos  # noqa: E402
from method.train import DEFAULT_SEEDS, train_do  # noqa: E402

RESULTS = ROOT / "notes" / "hardening" / "results"
BUILD = ROOT / "eval" / "test" / "build_noisy_benchmark.py"
ORDER = ["p020", "p010", "p005"]
MARGIN = 5.0

spec = importlib.util.spec_from_file_location("public_eval", ROOT / "eval" / "train" / "evaluate.py")
public_eval = importlib.util.module_from_spec(spec)
sys.modules["public_eval"] = public_eval  # required by @dataclass during module execution
spec.loader.exec_module(public_eval)


def finalize(level: str) -> None:
    subprocess.run([sys.executable, str(BUILD), "finalize", level], check=True)


def mean_f1(pairs):
    return float(np.mean([public_eval.compute_precision_recall_f1(gt, pred)[2] for gt, pred in pairs]))


def main() -> None:
    torch.set_num_threads(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"rule": __doc__.strip(), "levels": {}}
    chosen = None
    for level in ORDER:
        finalize(level)
        data = load_noisy_demos("train")
        do_pairs, count_pairs, betas = [], [], []
        for sd in data:
            n = sd.gt_node_count
            for seed in DEFAULT_SEEDS:
                run = train_do(sd, seed=seed)
                do_pairs.append((sd.gt_adj_matrix, public_eval.post_process_pred_adj(run["pred_core"], n)))
            betas.append(run["beta"])
            count_pairs.append((sd.gt_adj_matrix, public_eval.count_based_graph(sd.sequences, n)))
        do_f1, count_f1 = mean_f1(do_pairs), mean_f1(count_pairs)
        report["levels"][level] = {"reference_do_f1_train": do_f1, "count_based_f1_train": count_f1,
                                   "margin": do_f1 - count_f1,
                                   "procedures_with_beta_0.5": int(sum(b == 0.5 for b in betas))}
        print(level, report["levels"][level], flush=True)
        if do_f1 - count_f1 >= MARGIN:
            chosen = level
            break
    if chosen is None:
        chosen = ORDER[-1]
        report["note"] = "no level met the margin; the mildest level is used"
    finalize(chosen)
    report["chosen"] = chosen
    with open(ROOT / "notes" / "hardening" / "noise_level_selection.json", "w") as f:
        json.dump(report, f, indent=2)
    print("chosen noise level:", chosen)


if __name__ == "__main__":
    main()
