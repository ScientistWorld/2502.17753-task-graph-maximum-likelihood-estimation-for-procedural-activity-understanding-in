#!/bin/bash
# Setup script for TGML paper reproduction.
# Python packages are already installed at /home/user/.local/lib/python3.12/site-packages/
# and available via PYTHONPATH. No additional installation needed.

set -e
echo "setup.sh: Python packages from mounted workspace user site"
python3 -c "import torch; import numpy; import networkx; print('Packages available: torch', torch.__version__, 'numpy', numpy.__version__, 'networkx', networkx.__version__)"
