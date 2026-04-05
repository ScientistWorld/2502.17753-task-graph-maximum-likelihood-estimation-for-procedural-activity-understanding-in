#!/bin/bash
# Install Python packages into the container overlay.
#
set -e

# Install PyTorch with CUDA 12.1 support
uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install scientific computing packages
uv pip install --system \
    numpy scipy scikit-learn networkx \
    matplotlib pandas

# Install the paper's library from the cloned repo
cd /home/user/method/repo/lib
uv pip install --system -e .
