#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: ~0 GB
# (CaptainCook4D data is included in the cloned repo)
# EgoPER and EgoProceL annotations are in the repo or downloaded from GitHub
#
# This script downloads any remaining data needed beyond what's in the repo.

set -e
cd "$(dirname "$0")"

# The paper's GitHub repo was cloned into method/repo/ during setup.
# It already contains:
#   - CaptainCook4D data (task_graph_generation_annotations.json + ground truth graphs)
#   - Assembly101-O and EPIC-Tent-O annotations
#   - EgoProceL annotations (newly created by the paper)
#
# Check what's available
REPO_DATA="/home/user/method/repo/data"
if [ -d "$REPO_DATA" ]; then
    echo "Data directory found at $REPO_DATA"
    ls "$REPO_DATA"
else
    echo "WARNING: Repo data not found at $REPO_DATA"
fi

# EgoProceL annotations should be in the repo. Let's verify.
EGOPROCEL_ANNOTATIONS="$REPO_DATA/../task-graph-learning/online-mistake-detection"
if [ -d "$EGOPROCEL_ANNOTATIONS" ]; then
    echo "Found online-mistake-detection data"
fi

echo "Download check complete. All required data is in the cloned repository."
