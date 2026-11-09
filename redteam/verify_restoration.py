"""Read-only audit of restoration, excluding recorded initial dirty files."""
import hashlib
import json
import stat
import subprocess
from pathlib import Path

ROOT = Path('/home/user')
OUT = ROOT / 'redteam'
start = (OUT / 'start_commit.txt').read_text().strip()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


initial_dirty = []
for line in (OUT / 'initial_status.txt').read_text().splitlines():
    path = line[3:]
    if not path.startswith(('redteam/', '.git-shims/')):
        initial_dirty.append(path)
exclusions = ['redteam', 'REDTEAM.md', 'action.yaml', '.git-shims', *initial_dirty]
changes = git('diff', '--name-only', start, '--', '.',
              *[':(exclude)' + p for p in exclusions]).decode().splitlines()
mutation_checks = []
for backup in sorted(OUT.glob('*/_restore/mutation_*.json')):
    record = json.loads(backup.read_text())
    path = Path(record['path'])
    expected = record['content'].encode()
    mutation_checks.append({'path': str(path.relative_to(ROOT)),
                            'matches_backup': path.read_bytes() == expected})
score_checks = {}
for name in ['scores.json', 'scores_train.json', 'scores_test.json', 'reference.json']:
    path = ROOT / 'scoring' / name
    score_checks[name] = path.read_bytes() == git('show', f'{start}:scoring/{name}')
evaluator_checks = {}
for path in [ROOT / 'scripts/evaluate_test.sh', *sorted((ROOT / 'eval').rglob('*.py'))]:
    rel = str(path.relative_to(ROOT))
    evaluator_checks[rel] = path.read_bytes() == git('show', f'{start}:{rel}')
runtime = json.loads((OUT / 'stock_runtime/result.json').read_text())
report = {
    'start_commit': start,
    'unexpected_tracked_changes': changes,
    'preserved_initial_dirty_paths': initial_dirty,
    'mounted_devices_preserved': {
        p: stat.S_ISCHR((ROOT / p).stat().st_mode)
        for p in ['README.md', 'manifest.json']},
    'scripts_run_restored': (ROOT / 'scripts/run.sh').read_bytes() == git('show', f'{start}:scripts/run.sh'),
    'score_files_restored': score_checks,
    'evaluator_files_unchanged': evaluator_checks,
    'mutated_data_and_baseline_restored': mutation_checks,
    'temporary_networkx_source_absent': not (ROOT / 'environment/pylibs/networkx/algorithms/core.py').exists(),
    'stock_no_argument_test_exit_code': runtime['exit_code'],
    'protocol_exception': 'action.yaml must signal done; original unrelated dirty paths are preserved',
}
report['passed'] = (not changes and report['scripts_run_restored']
    and all(score_checks.values()) and all(evaluator_checks.values())
    and all(x['matches_backup'] for x in mutation_checks)
    and all(report['mounted_devices_preserved'].values())
    and report['temporary_networkx_source_absent'])
(OUT / 'restoration.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
assert report['passed'], 'gym restoration differs from the audit start'
