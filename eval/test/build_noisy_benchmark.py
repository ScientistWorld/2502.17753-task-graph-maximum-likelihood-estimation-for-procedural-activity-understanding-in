#!/usr/bin/env python3
"""Build the noisy-demonstration task-graph benchmark (hidden-test side).

Lives in eval/test/ because it can regenerate the hidden test graphs. Pure Python
(standard library only), deterministic given SECRET_SEED.

Graphs: random procedure DAGs calibrated to CaptainCook4D ground-truth graphs
(7-25 key-steps, ~1.23 edges per node, depth ~0.65 n): key-steps follow a latent
order; each step depends on 1-3 earlier steps (biased towards recent ones) or,
occasionally, only on START; the graph is transitively reduced and node ids are
randomly permuted (0 = START, n-1 = END).
Demonstrations: 4-10 per procedure, each a random topological sort (uniform
choice among the currently executable steps), then perturbed with the paper's
noise model (Section IV-F3): every key-step is, with probability p, replaced by a
random other step, deleted, or preceded by an inserted random step (uniform over
the three); repetitions are then removed keeping the first occurrence (paper
Section III-C1).

Commands:
  python3 eval/test/build_noisy_benchmark.py candidates
      writes eval/test/noisy_candidates/p{005,010,020}.json (train + test splits,
      distinct graphs per level)
  python3 eval/test/build_noisy_benchmark.py finalize P
      installs level P: data/noisy_demos/train.json (public, with graphs),
      data/noisy_demos/test_inputs.json (public, demonstrations only) and
      eval/test/noisy_demos_test.json (hidden graphs + demonstrations)
"""

import functools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CANDIDATE_DIR = HERE / "noisy_candidates"
SECRET_SEED = 0x6B1E3F0A9C2D47E58813D2A4F0B7C9E1
LEVELS = {"p005": 0.05, "p010": 0.10, "p020": 0.20}
N_PER_SPLIT = 40
GENERATOR = {"steps": [7, 25], "demonstrations": [4, 10], "p_extra_precondition": 0.28,
             "p_root_step": 0.06, "recency_decay": 0.4, "max_source_to_sink_paths": 2000}


def transitive_reduce(pre):
    @functools.lru_cache(None)
    def ancestors(v):
        out = set()
        for p in pre[v]:
            out |= {p} | ancestors(p)
        return frozenset(out)

    reduced = {v: set(pre[v]) for v in pre}
    for v in pre:
        for p in list(pre[v]):
            if any(p in ancestors(q) for q in pre[v] if q != p):
                reduced[v].discard(p)
    return reduced


def count_paths(n, pre):
    @functools.lru_cache(None)
    def paths(v):  # paths from v down to START following preconditions
        return 1 if v == 0 else sum(paths(p) for p in pre[v])
    return paths(n - 1)


def make_graph(rng, k):
    g = GENERATOR
    pre = {i: set() for i in range(1, k + 1)}
    for i in range(2, k + 1):
        if rng.random() < g["p_root_step"]:
            continue
        m = 1
        while m < 3 and rng.random() < g["p_extra_precondition"]:
            m += 1
        for _ in range(m):
            avail = [c for c in range(1, i) if c not in pre[i]]
            if not avail:
                break
            weights = [g["recency_decay"] ** (i - c - 1) for c in avail]
            pre[i].add(rng.choices(avail, weights=weights)[0])
    pre = transitive_reduce(pre)
    n = k + 2
    full = {0: set()}
    for i in range(1, k + 1):
        full[i] = set(pre[i]) if pre[i] else {0}
    has_successor = {p for v in range(1, k + 1) for p in full[v]}
    full[n - 1] = {v for v in range(1, k + 1) if v not in has_successor}
    perm = list(range(1, k + 1))
    rng.shuffle(perm)
    relabel = {0: 0, n - 1: n - 1}
    relabel.update({i: perm[i - 1] for i in range(1, k + 1)})
    return n, {relabel[v]: {relabel[p] for p in full[v]} for v in full}


def topological_sample(rng, n, pre):
    done = {0}
    remaining = set(range(1, n - 1))
    seq = []
    while remaining:
        available = sorted(v for v in remaining if pre[v] <= done)
        v = rng.choice(available)
        seq.append(v)
        done.add(v)
        remaining.discard(v)
    return seq


def perturb(rng, seq, n, p):
    steps = list(range(1, n - 1))
    out = []
    for v in seq:
        if rng.random() < p:
            kind = rng.randrange(3)
            if kind == 0:
                out.append(rng.choice([s for s in steps if s != v]))
            elif kind == 1:
                pass
            else:
                out.append(rng.choice(steps))
                out.append(v)
        else:
            out.append(v)
    dedup = []
    for v in out:
        if v not in dedup:
            dedup.append(v)
    return dedup


def build_split(rng, prefix, p):
    procedures = {}
    i = 0
    while len(procedures) < N_PER_SPLIT:
        k = rng.randint(*GENERATOR["steps"])
        n, pre = make_graph(rng, k)
        if count_paths(n, pre) > GENERATOR["max_source_to_sink_paths"]:
            continue
        d = rng.randint(*GENERATOR["demonstrations"])
        demos = [perturb(rng, topological_sample(rng, n, pre), n, p) for _ in range(d)]
        edges = sorted([pp, v] for v in pre for pp in pre[v])  # [a, b]: a is a precondition of b
        procedures[f"{prefix}_{i:02d}"] = {"n_nodes": n, "edges": edges, "demonstrations": demos}
        i += 1
    return procedures


def candidates():
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for name, p in LEVELS.items():
        rng = random.Random(f"{SECRET_SEED:x}-{name}")
        blob = {"noise_rate": p, "generator": GENERATOR,
                "train": build_split(rng, "noisytrain", p),
                "test": build_split(rng, "noisytest", p)}
        with open(CANDIDATE_DIR / f"{name}.json", "w") as f:
            json.dump(blob, f)
        print(name, "written")


def finalize(level):
    with open(CANDIDATE_DIR / f"{level}.json") as f:
        blob = json.load(f)
    p = blob["noise_rate"]
    data_dir = WORKSPACE / "data" / "noisy_demos"
    data_dir.mkdir(parents=True, exist_ok=True)
    note = ("Semi-synthetic noisy-demonstration task-graph benchmark (procedure graphs calibrated to "
            "CaptainCook4D, demonstrations perturbed with the paper's insert/delete/replace model). "
            "Node 0 = START, node n-1 = END, key-steps 1..n-2. Edge [a, b]: a is a precondition of b.")
    with open(data_dir / "train.json", "w") as f:
        json.dump({"noise_rate": p, "note": note, "procedures": blob["train"]}, f, indent=1)
    with open(data_dir / "test_inputs.json", "w") as f:
        json.dump({"noise_rate": p, "note": note + " Held-out procedures: graphs are not public.",
                   "procedures": {k: {"n_nodes": v["n_nodes"], "demonstrations": v["demonstrations"]}
                                  for k, v in blob["test"].items()}}, f, indent=1)
    hidden = {"noise_rate": p, "generator": blob["generator"],
              "edge_convention": "JSON edge [a, b] -> adjacency[b, a] = 1",
              "procedures": {}}
    for key, v in blob["test"].items():
        n = v["n_nodes"]
        hidden["procedures"][key] = {
            "name": key, "n_nodes": n, "edges": v["edges"],
            "sequences": [[0, 1] + [s + 1 for s in demo] + [n, n + 1] for demo in v["demonstrations"]],
        }
    with open(HERE / "noisy_demos_test.json", "w") as f:
        json.dump(hidden, f, indent=1)
    print(f"installed level {level} (p={p})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "candidates":
        candidates()
    elif len(sys.argv) >= 3 and sys.argv[1] == "finalize":
        finalize(sys.argv[2])
    else:
        raise SystemExit(__doc__)
