#!/bin/bash
# Shared Python environment setup for all shell entrypoints.
# Keep the broken mounted /home/user/.local tree off sys.path and rely on
# clean overlay-installed packages plus the small vendored pylibs directory.

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/user/environment/pylibs:/home/user:/usr/local/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages"
