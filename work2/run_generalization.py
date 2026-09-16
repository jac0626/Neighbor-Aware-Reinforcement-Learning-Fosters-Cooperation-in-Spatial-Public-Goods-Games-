"""Frozen-model development suites from GENERALIZATION_PROTOCOL.md."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
from engine import Config, METRICS, simulate
from run_pilot import run_one


def conditions(suite):
    cells = []
    if suite == 'messages':
        for r, attack in itertools.product([4.4, 4.8], [
                'none', 'high_defect', 'high_only', 'flip_action',
                'random_message', 'burst_defect', 'moderate_defect', 'high_cooperate']):
            cells.append({'r': r, 'rho': 0. if attack == 'none' else .1, 'attack': attack})
    elif suite == 'parameters':
        for r, rho in itertools.product([3., 3.6, 4., 4.2, 4.6], [0., .1]):
            cells.append({'r': r, 'rho': rho, 'attack': 'high_defect' if rho else 'none'})
        for rho in [.05, .2, .3, .5]:
            cells.append({'r': 4.8, 'rho': rho, 'attack': 'high_defect'})
    elif suite == 'scale':
        for side, rho in itertools.product([20, 50, 100], [0., .1]):
            cells.append({'L': side, 'r': 4.8, 'rho': rho,
                          'attack': 'high_defect' if rho else 'none'})
    elif suite == 'state':
        for (state, radius), r, rho in itertools.product(
                [('reputation', 1), ('own_action', 1), ('own_action', 2)],
                [4.4, 4.8], [0., .1]):
            cells.append({'state_mode': state, 'M': radius, 'r': r, 'rho': rho,
                          'attack': 'high_defect' if rho else 'none'})
    else:
        raise ValueError(f'Unknown suite: {suite}')
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--suite', choices=['messages', 'parameters', 'scale', 'state'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[216, 217, 218, 219])
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=96)
    args = parser.parse_args()
    if args.workers < 1 or not all(216 <= s <= 219 for s in args.seeds):
        parser.error('Use positive workers and reserved generalization-development seeds 216..219')
    selection = json.loads((args.validation / 'selection.json').read_text())
    origin = json.loads((args.validation / 'manifest.json').read_text())
    done = json.loads((args.validation / 'completion.json').read_text())
    assert done['completed_runs'] == len(origin['configs'])
    wanted = [selection['selected_by_family'][f] for f in ['learned', 'cooperation_gate']]
    wanted += ['iql-0', 'ni_global-0.5', 'trimmed_local-0.5', 'cooperation_first-0.2']
    models = [next(m for m in origin['models'] if m['label'] == label) for label in wanted]
    configs, labels = [], []
    for model, cell, seed in itertools.product(models, conditions(args.suite), args.seeds):
        configs.append(Config(**cell, seed=seed, steps=args.steps, method=model['method'],
                              kappa=model['kappa'], gate_scale=model['gate_scale'],
                              controller=tuple(model['controller']) if model['controller'] else None))
        labels.append(model['label'])
    # Validate every configuration before creating an output archive.
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    for name in ['engine.py', 'run_pilot.py', 'run_generalization.py',
                 'requirements.lock.txt', 'GENERALIZATION_PROTOCOL.md']:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    shutil.copyfile(args.validation / 'selection.json', source / 'model-selection.json')
    shutil.copyfile(args.validation / 'manifest.json', source / 'model-validation-manifest.json')
    manifest = {'purpose': 'generalization development validation, not final test',
                'suite': args.suite, 'models': models, 'configs': [asdict(c) for c in configs],
                'labels': labels, 'workers': min(args.workers, len(configs)),
                'metric_columns': ['step', *METRICS],
                'model_selection_path': str(args.validation.resolve()),
                'model_selection_rule': 'each family winner frozen before all generalization suites',
                'sources': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    simulate(Config(L=5, steps=100, record_every=100))
    tasks = [(i, c, str(args.out)) for i, c in enumerate(configs)]
    np.random.default_rng(913).shuffle(tasks)
    results, start = [], time.perf_counter()
    with mp.get_context('spawn').Pool(manifest['workers']) as pool:
        for i, result in pool.imap_unordered(run_one, tasks):
            results.append((i, result))
            if len(results) % 16 == 0 or len(results) == len(tasks):
                print(f'{len(results)}/{len(tasks)} {args.suite} runs; '
                      f'{time.perf_counter()-start:.1f}s', flush=True)
    fields = ['run', 'label', 'method', 'kappa', 'L', 'M', 'state_mode', 'r', 'rho',
              'attack', 'seed', 'steps', 'elapsed_seconds', 'whole_cooperation', *METRICS]
    with (args.out / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, result in sorted(results):
            row = {k: result['config'][k] for k in fields if k in result['config']}
            row.update(result['tail'])
            row.update(run=i, label=labels[i], elapsed_seconds=result['elapsed_seconds'],
                       whole_cooperation=result['whole']['cooperation'])
            writer.writerow(row)
    completion = {'completed_runs': len(results), 'wall_seconds': time.perf_counter()-start}
    (args.out / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(completion, flush=True)


if __name__ == '__main__':
    main()
