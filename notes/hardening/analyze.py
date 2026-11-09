#!/usr/bin/env python3
"""Hardening audit analysis: standardized rank deltas, paired comparisons, noise.

Rank rule (as used to rank attempts on this gym): for every experiment e and every
scored metric m in scoring/reference.json, improvement over the reference in the
metric's better direction, times |coefficient|, divided by the metric's standard
deviation across the scored population, summed over metrics and weighted by the
experiment weight. Here the population is every non-failed row of the audit for
that experiment. Failed rows sit at the floor of every metric.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
R = ROOT / "notes" / "hardening" / "results"
REFERENCE = "do"


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    spec = load(ROOT / "scoring" / "reference.json")["experiments"]
    test = load(R / "test_combined.json")
    train = load(R / "train_combined.json")
    details = load(R / "test_combined_details.json")["per_procedure"]
    case_files = sorted(glob.glob(str(R / "case_*.test.json")))

    out = {"experiments": {}, "combined": {}}
    combined = {}
    rng = np.random.default_rng(0)
    for exp, exp_spec in spec.items():
        if exp not in test["experiments"]:
            continue
        metrics = list(exp_spec["metrics"])
        rows = {k: dict(v) for k, v in test["experiments"][exp]["results"].items()}
        status = dict(test["method_status"][exp])
        diagnostics = dict(test["diagnostics"][exp])
        for path in case_files:
            case = os.path.basename(path)[len("case_"): -len(".test.json")]
            data = load(path)
            rows[case] = data["experiments"][exp]["results"][case]
            status[case] = data["method_status"][exp][case]
            diagnostics[case] = data["diagnostics"][exp].get(case, {})
            case_details = Path(path[: -len(".test.json")] + ".details.json")
            if case_details.is_file():
                extra = load(case_details)["per_procedure"][exp]
                if case in extra:
                    details[exp][case] = extra[case]
            ref_row = data["experiments"][exp]["results"][REFERENCE]
            assert all(abs(ref_row[m] - rows[REFERENCE][m]) < 1e-9 for m in metrics), (exp, case)
        train_rows = train["experiments"][exp]["results"]

        ok = [k for k in rows if status.get(k, {}).get("status") == "ok"]
        pop = {m: float(np.std([rows[k][m] for k in ok])) for m in metrics}
        ref = rows[REFERENCE]
        procs = sorted(details[exp][REFERENCE])
        ref_f1 = np.array([details[exp][REFERENCE][p]["f1"] for p in procs])
        boot = rng.integers(0, len(procs), size=(10000, len(procs)))

        table = {}
        for k, row in rows.items():
            parts = {}
            for m in metrics:
                sign = 1.0 if exp_spec["metrics"][m]["higher_is_better"] else -1.0
                coef = abs(exp_spec["metrics"][m]["coefficient"])
                parts[m] = coef * sign * (row[m] - ref[m]) / pop[m] if pop[m] > 0 else 0.0
            entry = {
                "status": status.get(k, {}).get("status"),
                "reason": status.get(k, {}).get("reason"),
                "test": row,
                "train_f1": train_rows.get(k, {}).get("f1"),
                "delta_f1": row["f1"] - ref["f1"],
                "delta_rank": float(sum(parts.values())),
                "rank_parts": parts,
                "trial_std_f1": diagnostics.get(k, {}).get("trial_std_f1"),
            }
            if k in details[exp] and k != REFERENCE:
                diffs = np.array([details[exp][k][p]["f1"] for p in procs]) - ref_f1
                means = diffs[boot].mean(axis=1)
                entry["paired"] = {
                    "mean_delta_f1": float(diffs.mean()),
                    "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                    "procedures_better": int((diffs > 1e-9).sum()),
                    "procedures_worse": int((diffs < -1e-9).sum()),
                }
            table[k] = entry
            combined.setdefault(k, 0.0)
            combined[k] += exp_spec["weight"] * entry["delta_rank"]

        reruns = [k for k in rows if k.startswith("do_rerun_seedset_")]
        noise = {
            "rerun_delta_f1": {k: table[k]["delta_f1"] for k in reruns},
            "rerun_delta_rank": {k: table[k]["delta_rank"] for k in reruns},
            "rerun_paired_ci95": {k: table[k].get("paired", {}).get("ci95") for k in reruns},
            "max_abs_rerun_delta_f1": max(abs(table[k]["delta_f1"]) for k in reruns) if reruns else None,
            "max_abs_rerun_delta_rank": max(abs(table[k]["delta_rank"]) for k in reruns) if reruns else None,
            "reference_trial_std_f1": diagnostics.get(REFERENCE, {}).get("trial_std_f1"),
        }
        out["experiments"][exp] = {"weight": exp_spec["weight"], "population_std": pop, "population": ok,
                                   "noise": noise, "methods": table}
    out["combined"] = combined

    with open(R / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = []
    for exp, block in out["experiments"].items():
        table = block["methods"]
        lines.append(f"### {exp} (weight {block['weight']})")
        lines.append("| method | status | test F1 | dF1 | train F1 | paired dF1 95% CI | better/worse procs | d rank (exp) |")
        lines.append("|---|---|---:|---:|---:|---|---|---:|")
        for k in sorted(table, key=lambda k: -table[k]["delta_rank"]):
            e = table[k]
            p = e.get("paired")
            ci = f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]" if p else ""
            bw = f"{p['procedures_better']}/{p['procedures_worse']}" if p else ""
            tr = f"{e['train_f1']:.2f}" if isinstance(e["train_f1"], (int, float)) else ""
            lines.append(f"| {k} | {e['status']} | {e['test']['f1']:.2f} | {e['delta_f1']:+.2f} | {tr} | {ci} | {bw} | "
                         f"{e['delta_rank']:+.3f} |")
        lines.append("")
        lines.append("Noise: " + json.dumps(block["noise"]))
        lines.append("Population std: " + json.dumps(block["population_std"]))
        lines.append("")
    lines.append("### Combined weighted d rank")
    lines.append("| method | combined d rank |")
    lines.append("|---|---:|")
    for k in sorted(combined, key=lambda k: -combined[k]):
        lines.append(f"| {k} | {combined[k]:+.3f} |")
    (R / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
