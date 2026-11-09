#!/usr/bin/env python3
"""Hardening audit analysis: standardized rank deltas, paired comparisons, noise.

Rank rule (as used to rank attempts on this gym): for every scored metric m in
scoring/reference.json, improvement over the reference in the metric's better
direction, times |coefficient|, divided by the metric's standard deviation across
the scored population; summed. Here the population is every non-failed row of the
audit (variants, reruns, shortcuts that produced valid artifacts, baseline).
Failed rows sit at the floor of every metric and therefore below every valid row.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
R = ROOT / "notes" / "hardening" / "results"
EXP = "captaincook4d_task_graph"
REFERENCE = "do"


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ref_spec = load(ROOT / "scoring" / "reference.json")["experiments"][EXP]["metrics"]
    metrics = list(ref_spec)

    test = load(R / "test_combined.json")
    rows = {k: dict(v) for k, v in test["experiments"][EXP]["results"].items()}
    status = dict(test["method_status"])
    diagnostics = dict(test.get("diagnostics", {}))
    for path in sorted(glob.glob(str(R / "sc_*.test.json"))):
        case = os.path.basename(path)[: -len(".test.json")]
        data = load(path)
        rows[case] = data["experiments"][EXP]["results"][case]
        status[case] = data["method_status"][case]
        if case in data.get("diagnostics", {}):
            diagnostics[case] = data["diagnostics"][case]
        # the reference row must be unaffected by the tampering
        ref_row = data["experiments"][EXP]["results"][REFERENCE]
        assert all(abs(ref_row[m] - rows[REFERENCE][m]) < 1e-9 for m in metrics), case

    train = load(R / "train_combined.json")["experiments"][EXP]["results"]
    details = load(R / "test_combined_details.json")["per_procedure"]

    ok = [k for k in rows if status.get(k, {}).get("status") == "ok"]
    pop = {m: float(np.std([rows[k][m] for k in ok])) for m in metrics}
    ref = rows[REFERENCE]

    def delta_rank(row):
        total = 0.0
        parts = {}
        for m in metrics:
            if pop[m] <= 0:
                parts[m] = 0.0
                continue
            sign = 1.0 if ref_spec[m]["higher_is_better"] else -1.0
            parts[m] = abs(ref_spec[m]["coefficient"]) * sign * (row[m] - ref[m]) / pop[m]
            total += parts[m]
        return total, parts

    rng = np.random.default_rng(0)
    procs = sorted(details[REFERENCE])
    ref_f1 = np.array([details[REFERENCE][p]["f1"] for p in procs])
    boot_idx = rng.integers(0, len(procs), size=(10000, len(procs)))

    table = {}
    for k in rows:
        dr, parts = delta_rank(rows[k])
        entry = {
            "status": status.get(k, {}).get("status"),
            "reason": status.get(k, {}).get("reason"),
            "test": rows[k],
            "train_f1": train.get(k, {}).get("f1"),
            "delta_f1": rows[k]["f1"] - ref["f1"],
            "delta_rank": dr,
            "rank_parts": parts,
            "trial_std_f1": diagnostics.get(k, {}).get("trial_std_f1"),
        }
        if k in details and k != REFERENCE:
            diffs = np.array([details[k][p]["f1"] for p in procs]) - ref_f1
            means = diffs[boot_idx].mean(axis=1)
            entry["paired"] = {
                "mean_delta_f1": float(diffs.mean()),
                "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "procedures_better": int((diffs > 1e-9).sum()),
                "procedures_worse": int((diffs < -1e-9).sum()),
            }
        table[k] = entry

    reruns = [k for k in rows if k.startswith("do_rerun_seedset_")]
    noise = {
        "rerun_delta_f1": {k: table[k]["delta_f1"] for k in reruns},
        "rerun_delta_rank": {k: table[k]["delta_rank"] for k in reruns},
        "max_abs_rerun_delta_f1": max(abs(table[k]["delta_f1"]) for k in reruns) if reruns else None,
        "max_abs_rerun_delta_rank": max(abs(table[k]["delta_rank"]) for k in reruns) if reruns else None,
        "reference_trial_std_f1": diagnostics.get(REFERENCE, {}).get("trial_std_f1"),
        "reference_trial_mean_f1": diagnostics.get(REFERENCE, {}).get("trial_mean_f1"),
    }
    if noise["reference_trial_std_f1"] is not None:
        noise["std_of_5_trial_mean_f1"] = noise["reference_trial_std_f1"] / np.sqrt(5)
    out = {"population_std": pop, "population": ok, "noise": noise, "methods": table}
    with open(R / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)

    order = sorted(table, key=lambda k: -table[k]["delta_rank"])
    lines = ["| method | status | test F1 | dF1 | train F1 | paired dF1 95% CI | better/worse procs | d rank |",
             "|---|---|---:|---:|---:|---|---|---:|"]
    for k in order:
        e = table[k]
        p = e.get("paired")
        ci = f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]" if p else ""
        bw = f"{p['procedures_better']}/{p['procedures_worse']}" if p else ""
        tr = f"{e['train_f1']:.2f}" if isinstance(e["train_f1"], (int, float)) else ""
        lines.append(f"| {k} | {e['status']} | {e['test']['f1']:.2f} | {e['delta_f1']:+.2f} | {tr} | {ci} | {bw} | "
                     f"{e['delta_rank']:+.3f} |")
    lines.append("")
    lines.append("Noise: " + json.dumps({k: v for k, v in noise.items() if k != "reference_trial_mean_f1"}))
    lines.append("Population std: " + json.dumps(pop))
    (R / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
