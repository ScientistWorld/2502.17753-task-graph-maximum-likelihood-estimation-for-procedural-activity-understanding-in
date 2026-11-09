#!/bin/bash
# Shared Python environment setup for all shell entrypoints.
# Packages (numpy, networkx, torch) come from the container (environment/setup.sh);
# no vendored copies are put on the path.

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
unset PYTHONHOME
unset PYTHONUSERBASE
export PYTHONPATH="/home/user:/usr/local/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages"
