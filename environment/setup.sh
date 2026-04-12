#!/bin/bash
# Install a clean CPU Python stack into the container overlay.
# Compute jobs must not depend on the broken mounted /home/user/.local tree.
set -e

export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1

uv pip install --system \
  numpy==2.1.3 \
  networkx==3.4.2 \
  torch==2.5.1

echo "setup.sh: installed clean system numpy/networkx/torch"
