"""Record the literal stock no-argument test result with no prerequisite repair."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/user')
OUT = ROOT / 'redteam/stock_runtime'
OUT.mkdir(parents=True, exist_ok=True)
paths = [ROOT / 'scoring' / p for p in ['scores.json', 'scores_train.json', 'scores_test.json']]
saved = {p: p.read_bytes() if p.exists() else None for p in paths}
assert not (ROOT / 'environment/pylibs/networkx/algorithms/core.py').exists()
try:
    with (OUT / 'test.log').open('w') as stream:
        result = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
    status = {'command': 'bash scripts/evaluate_test.sh',
              'exit_code': result.returncode, 'prerequisite_repair': False}
    (OUT / 'result.json').write_text(json.dumps(status, indent=2) + '\n')
    print(status, flush=True)
finally:
    for p, content in saved.items():
        if content is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(content)
