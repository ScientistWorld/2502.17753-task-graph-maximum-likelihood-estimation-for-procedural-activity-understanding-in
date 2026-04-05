#!/bin/bash
# Job script submitted via action.yaml.
# Runs inside the compute container with GPU(s) available.

set -e
cd /home/user

# Run the full reproduction pipeline
bash scripts/reproduce.sh
