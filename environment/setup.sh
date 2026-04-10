#!/bin/bash
# Setup script for TGML paper reproduction.
# Hardcode PYTHONPATH and LD_LIBRARY_PATH since %environment is not
# available during setup.sh execution.

set -e
export PYTHONPATH=/home/user/.local/lib/python3.12/site-packages:/home/user
export LD_LIBRARY_PATH=/home/user/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/home/user/.local/lib/python3.12/site-packages/nvidia/cublas/lib:/home/user/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/user/.local/lib/python3.12/site-packages/nvidia/cusparselt/lib:/home/user/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/home/user/.local/lib/python3.12/site-packages/nvidia/nvshmem/lib
export PYTHONDONTWRITEBYTECODE=1

echo "setup.sh: verifying Python packages from mounted workspace user site..."
python3 -c "import torch; import numpy; import networkx; print('Packages available: torch', torch.__version__, 'numpy', numpy.__version__, 'networkx', networkx.__version__)"
