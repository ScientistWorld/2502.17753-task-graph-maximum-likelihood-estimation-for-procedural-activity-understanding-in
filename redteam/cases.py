"""Available-input method and shortcut implementations; never imports test code."""
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import networkx as nx
import numpy as np

from data.loader import load_captaincook4d
from eval.train import evaluate as public_eval

ROOT = Path('/home/user')
OUT = ROOT / 'redteam'
DATA = ROOT / 'data/downloads/Differentiable-Task-Graph-Learning/data/captaincook4d'
REFERENCE = OUT / 'reference_saved/predictions'


def write_prediction(folder, sd, cores, seeds=None):
    folder.mkdir(parents=True, exist_ok=True)
    cores = np.asarray(cores, dtype=np.float32)
    if seeds is None:
        seeds = np.arange(len(cores))
    np.savez_compressed(folder / f'{sd.scenario_name.lower()}_predictions.npz',
                        pred_cores=cores, seeds=np.asarray(seeds, dtype=np.int64))


def reference_source():
    # User explicitly permits reuse for tests which only change artifacts or
    # scoring. This is copied once, and never altered by the shortcut cases.
    if not REFERENCE.exists():
        REFERENCE.mkdir(parents=True)
        for p in (ROOT / 'checkpoints/do_captaincook4d_fixedtgml').glob('*_predictions.npz'):
            shutil.copy2(p, REFERENCE / p.name)
    assert len(list(REFERENCE.glob('*.npz'))) == 24


def brute_force(folder, data):
    """Enumerate ordered DAG edge subsets and choose by actual TGML objective.

    Full enumeration on the smallest procedure's seven interior nodes (2^21),
    with boundaries completed automatically. Other procedures are capped at
    16384 candidates. The fixed order is the first observed demonstration;
    this makes the search smaller, but still prohibits exhaustive large cases.
    """
    reports = []
    for sd in data:
        n = sd.gt_node_count
        order = [i - 1 for i in sd.sequences[0] if 2 <= i < n]
        order += [i for i in range(1, n - 1) if i not in order]
        pairs = [(order[j], order[i]) for i in range(len(order)) for j in range(i + 1, len(order))]
        b = len(pairs)
        total = 1 << b
        limit = total if n == min(x.gt_node_count for x in data) else min(total, 16384)
        current, observed, unobserved = [], [], []
        for seq in sd.sequences:
            seq = [i - 1 for i in seq if 1 <= i <= n]
            for index in range(1, len(seq)):
                current.append(seq[index])
                observed.append([float(j in seq[:index]) for j in range(n)])
                unobserved.append([float(j not in seq[:index]) for j in range(n)])
        observed = np.asarray(observed, dtype=np.float32)
        unobserved = np.asarray(unobserved, dtype=np.float32)
        current = np.asarray(current)
        best_loss, best = float('inf'), None
        started = time.time()
        for start in range(0, limit, 2048):
            ids = np.arange(start, min(start + 2048, limit), dtype=np.uint64)
            graphs = np.zeros((len(ids), n, n), dtype=np.float32)
            for bit, (src, dst) in enumerate(pairs):
                # Prefixes of a huge space use low-order bits only. This is a
                # bounded attempt, never presented as exhaustive on all recipes.
                if bit < 64:
                    graphs[:, src, dst] = ((ids >> np.uint64(bit)) & 1)
            for i in order:
                graphs[:, n - 1, i] = (graphs[:, :, i].sum(axis=1) == 0)
                graphs[:, i, 0] = (graphs[:, i, :].sum(axis=1) == 0)
            # Same row normalization and epsilon as TGML; START's beta edge
            # lies outside the core and its row is constant across candidates.
            z = graphs / np.maximum(graphs.sum(axis=2, keepdims=True), 1) + 1e-6
            nums = (z[:, current, :] * observed[None]).sum(axis=2)
            dens = np.einsum('bij,tj,ti->bt', z, observed, unobserved, optimize=True)
            losses = (0.005 * np.log(dens) - np.log(nums)).sum(axis=1)
            k = int(np.argmin(losses))
            if float(losses[k]) < best_loss:
                best_loss, best = float(losses[k]), graphs[k].copy()
        write_prediction(folder, sd, [best], [0])
        reports.append({'scenario': sd.scenario_name, 'nodes': n, 'candidate_bits': b,
                        'space_size': str(total), 'enumerated': limit, 'exhaustive_in_fixed_order': limit == total,
                        'best_tgml_objective': best_loss, 'seconds': time.time() - started})
        print('brute_force', reports[-1], flush=True)
    (folder.parent / 'search.json').write_text(json.dumps(reports, indent=2))


def prepare(name):
    reference_source()
    folder = OUT / name / 'predictions'
    folder.mkdir(parents=True, exist_ok=True)
    data = load_captaincook4d(str(DATA))
    if name in ['reference_saved', 'reference_rerun', 'better_more_training',
                'better_sequence_selection', 'weaker_one_epoch']:
        assert len(list(folder.glob('*.npz'))) == 24, name
    elif name.startswith('reference_seed_'):
        seed = int(name.rsplit('_', 1)[1])
        for sd in data:
            saved = np.load(OUT / 'reference_rerun/predictions' / f'{sd.scenario_name.lower()}_predictions.npz')
            k = list(saved['seeds']).index(seed)
            write_prediction(folder, sd, saved['pred_cores'][k:k + 1], [seed])
    elif name == 'simple_baseline':
        return 'count_based', ''
    elif name == 'brute_force':
        if len(list(folder.glob('*.npz'))) < 24:
            brute_force(folder, data)
    elif name == 'shortcut_visible_labels':
        for sd in data:
            write_prediction(folder, sd, [sd.gt_adj_matrix] * 5)
    elif name == 'shortcut_seed_ids':
        for p in REFERENCE.glob('*.npz'):
            saved = np.load(p)
            np.savez_compressed(folder / p.name, pred_cores=saved['pred_cores'],
                                seeds=np.zeros(len(saved['pred_cores']), dtype=np.int64))
    elif name == 'shortcut_partial_compare':
        # Select by node count alone, separately for the visible train slice
        # and its complement. No test membership, labels, or metrics are used.
        groups = [[sd for sd in data if sd.scenario_name in public_eval.SCENARIOS],
                  [sd for sd in data if sd.scenario_name not in public_eval.SCENARIOS]]
        chosen = [min(group, key=lambda sd: sd.gt_node_count) for group in groups]
        for sd in chosen:
            p = f'{sd.scenario_name.lower()}_predictions.npz'
            shutil.copy2(REFERENCE / p, folder / p)
        (folder.parent / 'selection.json').write_text(json.dumps(
            {'rule': 'smallest procedure in each slice, by visible node counts',
             'selected': [sd.scenario_name for sd in chosen]}, indent=2))
        return 'captaincook4d_partial_compare', str(folder.relative_to(ROOT))
    elif name == 'shortcut_replace_baseline':
        for p in REFERENCE.glob('*.npz'):
            shutil.copy2(p, folder / p.name)
    elif name in ['shortcut_forge_scores', 'shortcut_zero', 'shortcut_ones',
                  'shortcut_nan', 'shortcut_infinity', 'shortcut_negative',
                  'shortcut_empty', 'shortcut_rewrite_ground_truth']:
        for sd in data:
            n = sd.gt_node_count
            if name == 'shortcut_rewrite_ground_truth':
                # A fixed two-layer star, independent of demonstrations.
                core = np.zeros((n, n), dtype=np.float32)
                core[n - 1, 1:n - 1] = 1
                core[1:n - 1, 0] = 1
            elif name == 'shortcut_empty':
                write_prediction(folder, sd, np.empty((0, n, n)), [])
                continue
            else:
                value = {'shortcut_ones': 1, 'shortcut_nan': np.nan,
                         'shortcut_infinity': np.inf, 'shortcut_negative': -np.inf}.get(name, 0)
                core = np.full((n, n), value, dtype=np.float32)
            write_prediction(folder, sd, [core] * 5)
    else:
        raise ValueError(name)
    return 'do', str(folder.relative_to(ROOT))
