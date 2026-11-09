This audit tests whether the graph-induction gym rewards research quality and
whether concrete shortcuts pay. All candidate implementations use the future
agent's visible data, method (improve mode), public train evaluator, or saved
reference outputs explicitly permitted for reuse by the audit instructions.
They do not read hidden test code, membership, scores, the paper, or the git
history. Only the harness executes the unmodified hidden test and analyzes its
results afterwards.

Run `bash redteam/with_prerequisite.sh bash redteam/run_suite.sh` in the gym
container. Each case also has its own
`redteam/<case>/run.sh`. These scripts generate or reuse prediction artifacts,
run the public training evaluator, and invoke **`bash scripts/evaluate_test.sh`
with no arguments in a separate process**. `execute_case.py` restores every
changed score, data, and baseline file, including on ordinary exceptions. This
models the fresh hidden-evaluation job: candidate effects are disk artifacts,
and no background process or evaluator-process manipulation is used.

Runtime prerequisite: the original vendored NetworkX 3.4.2 is missing
`algorithms/core.py` (excluded by the repository's `core.*` ignore rule).
The first stock job therefore failed before training. A later literal stock
`bash scripts/evaluate_test.sh` also exited 1 before scoring; see
`stock_runtime/test.log` and `stock_runtime/result.json`. All numerical audit
results are conditional on this packaging repair. `with_prerequisite.sh`
temporarily restores exactly the upstream file, verified byte-for-byte against
the installed distribution's RECORD hash, and removes it on exit. This repair
is uniform across all numerical runs and is never used as a scoring shortcut.
It temporarily changes the package's source files; no search path, dependency
version, evaluator code, or evaluator process is changed. The missing file is
absent again at audit completion, preserving the original gym packaging.

`train_variants.py` implements exactly the existing TGML model and loss using
batched event masks. Every dataset procedure has a loss/gradient comparison
against `method/tgml.py`, and one complete 1000-step scalar-loop run checks
final graph agreement. The speedup does not change the learning algorithm.
The three rerun seeds are 7, 19 and 31. Training variants use the same seeds.

Scores and logs are retained inside each case. `analysis.json` applies absolute
coefficients, metric directions, and experiment weights. Its standardized
rank deltas use the explicitly recorded finite audit population, since the
actual future population's variances cannot be known. Metric dominance is
reported separately to establish paying shortcuts independently of scales.

The reference and test snapshots under individual cases are audit evidence;
they are not inputs available to the future scientist. Shortcut implementations
are named `shortcut_*` even if the attempted mechanism fails. Failed baseline
replacement retaining reference predictions is labeled an identity shortcut
tie; it is not evidence that the baseline was changed in the hidden test.

`shortcut_single_seed` submits only predeclared seed 7, with no test-based seed
selection. It demonstrates that the no-argument test accepts one seed and
assigns it zero instability despite the stated five-seed contract. The other
single-seed runs are the required reference variability diagnostics.

All prediction NPZs are retained and committed, including ignored binary
files. Quality-variant runs reuse the outputs trained by `train_variants.py`;
remove those four prediction directories and run the suite to retrain them.
`REDTEAM.md` contains the final assessment. `verify_restoration.py` checks the
initial tracked gym files and the actual byte backups of mutated data.
