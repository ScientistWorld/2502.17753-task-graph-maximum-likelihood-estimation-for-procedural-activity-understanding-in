#!/bin/bash
# Fetch the public CaptainCook4D task-graph data released with the paper.
# Estimated download size: ~42 MB (repository tarball); disk usage after extraction: ~5 MB.
#
# Source: pinned commit of https://github.com/fpv-iplab/Differentiable-Task-Graph-Learning,
# downloaded as a tarball (no git history is kept) and reduced to data/captaincook4d/.
#
# The 12 held-out procedures scored by the hidden test keep only their node list
# ("steps", which defines the prediction indexing); their ground-truth edges are
# withheld here and live only in eval/test/.
#
# Idempotent: re-running re-verifies the data and re-applies the withholding step.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="536aaa72fca6b4a16de7ffde05e3864466986aa4"
URL="https://codeload.github.com/fpv-iplab/Differentiable-Task-Graph-Learning/tar.gz/${COMMIT}"
REPO_DIR="$ROOT_DIR/data/downloads/Differentiable-Task-Graph-Learning"
DEST="$REPO_DIR/data/captaincook4d"
ANNOTATIONS_SHA256="712a78bf63dd1b2a11b2b7a2417f3429586c3d8bca47adf6d2adb630fa50f07e"
HELD_OUT="blenderbananapancakes breakfastburritos buttercorncup coffee cucumberraita panfriedtofu ramen scrambledeggs spicedhotchocolate spicytunaavocadowraps tomatochutney zoodles"

if [ ! -f "$DEST/.prepared" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL --retry 3 -o "$tmp/repo.tar.gz" "$URL"
  tar -xzf "$tmp/repo.tar.gz" -C "$tmp" --wildcards "*/data/captaincook4d/*"
  rm -rf "$REPO_DIR"
  mkdir -p "$REPO_DIR/data"
  mv "$tmp/Differentiable-Task-Graph-Learning-${COMMIT}/data/captaincook4d" "$DEST"
fi

echo "${ANNOTATIONS_SHA256}  ${DEST}/task_graph_generation_annotations.json" | sha256sum -c -

python3 - "$DEST/ground_truth_task_graphs" $HELD_OUT <<'PY'
import json
import os
import sys

graph_dir = sys.argv[1]
for key in sys.argv[2:]:
    path = os.path.join(graph_dir, key + ".json")
    with open(path) as f:
        graph = json.load(f)
    if "edges" in graph:
        with open(path, "w") as f:
            json.dump({"steps": graph["steps"],
                       "edges_withheld": "held-out test procedure; ground truth is not public in this gym"},
                      f, indent=1)
            f.write("\n")
PY

touch "$DEST/.prepared"
echo "CaptainCook4D task-graph data ready in $DEST (held-out edges withheld)."
