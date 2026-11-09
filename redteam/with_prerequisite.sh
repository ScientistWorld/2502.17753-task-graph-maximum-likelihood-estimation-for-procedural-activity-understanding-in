#!/bin/bash
# Repair a missing upstream source file uniformly for the audit, then restore
# the original filesystem. This is a runtime prerequisite, never a shortcut.
set -euo pipefail
cd /home/user
target=environment/pylibs/networkx/algorithms/core.py
created=0
cleanup() {
    if [ "$created" -eq 1 ]; then
        rm -f "$target"
        rm -f environment/pylibs/networkx/algorithms/__pycache__/core.*.pyc
    fi
}
trap cleanup EXIT
if [ ! -f "$target" ]; then
    python3 - <<'PY'
import base64, csv, hashlib
from pathlib import Path
source = Path('redteam/dependencies/networkx_core.py').read_bytes()
record = next(r for r in csv.reader(open('environment/pylibs/networkx-3.4.2.dist-info/RECORD'))
              if r[0] == 'networkx/algorithms/core.py')
digest = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(source).digest()).decode().rstrip('=')
assert digest == record[1]
Path('environment/pylibs/networkx/algorithms/core.py').write_bytes(source)
PY
    created=1
fi
"$@"
