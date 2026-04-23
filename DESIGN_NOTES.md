# Design Notes - Train/Test Evaluation Improvements

## Changes Made in This Pass

### 1. Enhanced Metric Triangulation
Added three structural metrics to `reference.json` and implemented them in both `eval/train/evaluate.py` and `eval/test/evaluate.py`:

- **`edge_density_ratio`** (higher_is_better, coefficient: 0.15): Ratio of predicted edge count to GT edge count. Catches under/over-connection. A method that adds spurious edges to boost F1 will have density >> 1.0.

- **`path_coverage`** (higher_is_better, coefficient: 0.12): Fraction of GT directed paths covered by predicted graph. Measures structural correctness beyond edge overlap. A shortcut method that matches edges but wrong topology will have low path coverage.

- **`topological_distance`** (higher_is_better is false, coefficient: -0.18): Normalized topological ordering distance between predicted and GT DAGs. Rewards structurally similar graphs regardless of edge overlap. Catches methods that predict edges but with wrong dependency ordering.

### 2. Metric Rationale
These three metrics probe different aspects of the paper's core claim (graph quality):
- **edge_density_ratio**: Direct proxy for over/under-connection
- **path_coverage**: Structural correctness of path chains
- **topological_distance**: Dependency ordering correctness

A method that gamed F1 by adding spurious edges would fail on density_ratio AND path_coverage simultaneously. A method that got edge overlap right but wrong topology would fail on topological_distance. This multi-metric triangulation makes shortcut detection robust.

### 3. Test Slice Calibration
The test slice has 12 scenarios (same order of magnitude as train's 13). The DO method should score similarly on both slices since it's not overfitting — the gap is expected to be small (< 2% on F1). A shortcut method would regress more on test.

### 4. Leakage Check
- `eval/train/` and `eval/test/` are fully independent (no cross-imports)
- Both use identical metric computation code for fair comparison
- Scripts auto-discover methods from `checkpoints/` directory

## Design Considerations for Future Iterations

### Potential Improvements
1. **Threshold robustness metric**: Add a metric that varies the binarization threshold and measures how sensitive the graph quality is. A well-designed method should be stable across thresholds.

2. **Scenario difficulty stratification**: If scenarios have varying difficulty (e.g., different number of steps), stratify train/test to ensure similar difficulty distributions.

3. **Seed variance analysis**: The current `std_f1` measures scenario-to-scenario variance. Adding seed-to-seed variance analysis could detect unstable methods.

### Known Limitations
- The path coverage metric uses path enumeration which can be expensive for large graphs. The current implementation has graceful fallbacks on failure.
- Topological distance computation uses heuristic longest-path which may not be exact for all DAG structures. Falls back to ordinal ranking on failure.

## Validation
- Both `evaluate_train.sh` and `evaluate_test.sh` pass `bash -n` syntax check
- No cross-imports between train and test eval folders
- `reference.json` metrics have all required fields (higher_is_better, coefficient, description)
- Scripts auto-discover methods (no hardcoded defaults)

## Next Steps for Runner
1. Run `scripts/evaluate_train.sh` to produce `scoring/scores_train.json` with new metrics
2. Run `scripts/evaluate_test.sh` to produce `scoring/scores_test.json` with new metrics
3. Verify scores show small gap between train and test for DO method
4. Verify structural metrics (density, path_coverage, topo_distance) are present in output