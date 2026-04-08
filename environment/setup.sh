#!/bin/bash
# Install Python packages into the container overlay.
#
set -e

# The scratch fallback copies a working Python runtime from the host, including
# torch. Bootstrap pip from the copied stdlib because builder hosts do not
# expose a portable distro `pip` package layout.
python3 -m ensurepip --upgrade

# Install only the missing Python dependencies needed by this workspace.
python3 -m pip install --break-system-packages \
    scipy scikit-learn networkx matplotlib pandas pyyaml click

# Install the paper's library from the cloned repo
cd /home/user/method/repo/lib
python3 -m pip install --break-system-packages -e .
