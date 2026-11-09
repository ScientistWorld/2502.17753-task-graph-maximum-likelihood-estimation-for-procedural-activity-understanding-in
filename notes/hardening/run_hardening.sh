#!/bin/bash
# Hardening audit (job body). Requires the reference in checkpoints/do.
#   1. trains honest variants from the final code and builds shortcut submissions;
#   2. replays every case through the stock no-argument entry points
#      (scripts/evaluate_test.sh and scripts/evaluate_train.sh);
#   3. replays the cases that tamper with editable files one at a time, restoring
#      every touched file afterwards.
set -euo pipefail
cd /home/user
source /home/user/scripts/common_env.sh

H=notes/hardening
R=$H/results
GT_DIR=data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d/ground_truth_task_graphs
BACKUP="$(mktemp -d)"
MUTATION_CASES=" sc_rewrite_ground_truth sc_partial_coverage sc_forge_scores "
mkdir -p "$R"

cp -r "$GT_DIR" "$BACKUP/gt"
cp scoring/scores_train.json "$BACKUP/scores_train.json"
cp scoring/scores_test.json "$BACKUP/scores_test.json"
STAGED=()
restore() {
    for name in "${STAGED[@]:-}"; do
        [ -n "$name" ] && rm -rf "checkpoints/$name"
    done
    rm -rf "$GT_DIR" && cp -r "$BACKUP/gt" "$GT_DIR"
    cp "$BACKUP/scores_train.json" scoring/scores_train.json
    cp "$BACKUP/scores_test.json" scoring/scores_test.json
}
trap restore EXIT

echo "=== [hardening] training variants ($(date)) ==="
python3 $H/train_variants.py 2>&1 | tee "$R/train_variants.log"
echo "=== [hardening] building shortcut submissions ==="
python3 $H/build_shortcuts.py 2>&1 | tee "$R/build_shortcuts.log"

echo "=== [hardening] stage A: all prediction-only cases through the no-arg entry points ==="
for dir in $H/runs/*/; do
    name="$(basename "$dir")"
    case "$MUTATION_CASES" in *" $name "*) continue ;; esac
    rm -rf "checkpoints/$name"
    cp -r "$dir" "checkpoints/$name"
    STAGED+=("$name")
done
bash scripts/evaluate_test.sh > "$R/test_combined.log" 2>&1
cp scoring/scores_test.json "$R/test_combined.json"
python3 eval/test/evaluate.py --output "$BACKUP/rerun_test.json" --details-output "$R/test_combined_details.json" > /dev/null
cmp -s <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['experiments'],sort_keys=True))" "$R/test_combined.json") \
       <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['experiments'],sort_keys=True))" "$BACKUP/rerun_test.json") \
    && echo "test rescoring is deterministic" || echo "WARNING: test rescoring differs between runs"
bash scripts/evaluate_train.sh > "$R/train_combined.log" 2>&1
cp scoring/scores_train.json "$R/train_combined.json"
python3 eval/train/evaluate.py --output "$BACKUP/rerun_train.json" --details-output "$R/train_combined_details.json" > /dev/null
for name in "${STAGED[@]}"; do rm -rf "checkpoints/$name"; done
STAGED=()

run_case() {
    local name="$1"
    rm -rf "checkpoints/$name"
    cp -r "$H/runs/$name" "checkpoints/$name"
    STAGED=("$name")
    bash scripts/evaluate_test.sh > "$R/$name.test.log" 2>&1
    cp scoring/scores_test.json "$R/$name.test.json"
    rm -rf "checkpoints/$name"
    STAGED=()
}

echo "=== [hardening] stage B: sc_rewrite_ground_truth (public held-out graphs rewritten to the boundary star) ==="
python3 - "$GT_DIR" <<'PY'
import json, os, sys
graph_dir = sys.argv[1]
held_out = ["blenderbananapancakes", "breakfastburritos", "buttercorncup", "coffee", "cucumberraita",
            "panfriedtofu", "ramen", "scrambledeggs", "spicedhotchocolate", "spicytunaavocadowraps",
            "tomatochutney", "zoodles"]
for name in sorted(os.listdir(graph_dir)):
    path = os.path.join(graph_dir, name)
    graph = json.load(open(path))
    n = len(graph["steps"])
    # All-zero cores post-process to END->v and v->START for every step v.
    # JSON edge [a, b] encodes adjacency[b, a].
    graph["edges"] = [[v, n - 1] for v in range(1, n - 1)] + [[0, v] for v in range(1, n - 1)]
    graph.pop("edges_withheld", None)
    json.dump(graph, open(path, "w"))
print("rewrote", len(os.listdir(graph_dir)), "public graph files")
PY
run_case sc_rewrite_ground_truth
rm -rf "$GT_DIR" && cp -r "$BACKUP/gt" "$GT_DIR"

echo "=== [hardening] stage B: sc_partial_coverage (+ registry key captaincook4d_partial_compare) ==="
python3 - <<'PY'
import json
scores = json.load(open("scoring/scores_train.json"))
scores.setdefault("experiments", {}).setdefault("captaincook4d_task_graph", {}).setdefault("results", {})[
    "captaincook4d_partial_compare"] = {"f1": 100.0}
scores.setdefault("method_artifacts", {})["captaincook4d_partial_compare"] = {
    "artifact_type": "npz_prediction_directory", "checkpoint_dir": "checkpoints/sc_partial_coverage"}
json.dump(scores, open("scoring/scores_train.json", "w"), indent=2)
PY
run_case sc_partial_coverage
cp "$BACKUP/scores_train.json" scoring/scores_train.json

echo "=== [hardening] stage B: sc_forge_scores (forged score files written before scoring) ==="
python3 - <<'PY'
import json
forged = {"precision": 1e9, "recall": 1e9, "f1": 1e9, "worst_quartile_f1": 1e9,
          "edge_density_ratio": 1.0, "path_coverage": 1.0, "topological_distance": 0.0}
for path in ("scoring/scores_test.json", "scoring/scores_train.json"):
    scores = json.load(open(path))
    scores["experiments"]["captaincook4d_task_graph"]["results"]["sc_forge_scores"] = dict(forged)
    scores["experiments"]["captaincook4d_task_graph"]["results"]["do"] = {k: 0.0 for k in forged}
    scores["method_status"] = {"sc_forge_scores": {"status": "ok"}}
    json.dump(scores, open(path, "w"), indent=2)
PY
run_case sc_forge_scores
cp "$BACKUP/scores_train.json" scoring/scores_train.json

echo "=== [hardening] done ($(date)) ==="
