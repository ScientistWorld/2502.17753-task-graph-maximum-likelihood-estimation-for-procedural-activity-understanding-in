#!/bin/bash
# Install Python dependencies into the container environment.
# Keep this self-contained: compute nodes have no internet, but the scheduler
# runs setup.sh while building/updating the workspace overlay.
set -e

export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1

uv pip install --system \
  numpy \
  networkx==3.4.2 \
  torch

echo "setup.sh: installed numpy, networkx, torch"
