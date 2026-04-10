#!/bin/bash
# Setup script for TGML paper reproduction.
# Tries to import torch. If it fails (workspace not mounted yet), warn but don't fail.
# torch is either installed in %post or will come from the mounted workspace at job runtime.

export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/site-packages:/home/user/.local/lib/python3.12/site-packages:/home/user
export PYTHONDONTWRITEBYTECODE=1

echo "setup.sh: checking Python packages..."
if python3 -c "import torch; import numpy; import networkx; print('  torch', torch.__version__, 'numpy', numpy.__version__, 'networkx', networkx.__version__)" 2>/dev/null; then
    echo "  Packages from container installation: OK"
else
    echo "  WARNING: torch not found (workspace may not be mounted yet). Will rely on workspace at job runtime."
fi
echo "setup.sh: done."
