#!/bin/bash
# Install only the package that is broken in the mounted user site-packages.
# The workspace already carries working torch/numpy builds in /home/user/.local,
# so installing torch here would waste the 20-minute setup budget.
set -e

export UV_SYSTEM_PYTHON=1
export UV_BREAK_SYSTEM_PACKAGES=1

uv pip install --system \
  networkx==3.4.2

echo "setup.sh: installed clean system networkx"
