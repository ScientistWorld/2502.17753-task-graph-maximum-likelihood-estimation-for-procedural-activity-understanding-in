#!/bin/bash
# Fetch all public, pre-existing artifacts needed by the reproduction.
# Estimated download size: <100 MB; disk usage: <200 MB for the pinned paper repo data.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PAPER_REPO_URL="https://github.com/fpv-iplab/Differentiable-Task-Graph-Learning.git"
PAPER_REPO_COMMIT="536aaa72fca6b4a16de7ffde05e3864466986aa4"
PAPER_REPO_REF="main"
PAPER_REPO_DIR="$ROOT_DIR/data/downloads/Differentiable-Task-Graph-Learning"
CAPTAINCOOK_DIR="$PAPER_REPO_DIR/data/captaincook4d"

mkdir -p "$(dirname "$PAPER_REPO_DIR")"

if [ ! -d "$PAPER_REPO_DIR/.git" ]; then
  rm -rf "$PAPER_REPO_DIR"
  git clone --depth 1 --branch "$PAPER_REPO_REF" "$PAPER_REPO_URL" "$PAPER_REPO_DIR"
fi

git -C "$PAPER_REPO_DIR" remote set-url origin "$PAPER_REPO_URL"
git -C "$PAPER_REPO_DIR" fetch --depth 1 origin "$PAPER_REPO_REF"
git -C "$PAPER_REPO_DIR" checkout --detach "$PAPER_REPO_COMMIT"

test -f "$CAPTAINCOOK_DIR/task_graph_generation_annotations.json"
test -d "$CAPTAINCOOK_DIR/ground_truth_task_graphs"

echo "Downloaded paper repository and verified CaptainCook4D task-graph data."
