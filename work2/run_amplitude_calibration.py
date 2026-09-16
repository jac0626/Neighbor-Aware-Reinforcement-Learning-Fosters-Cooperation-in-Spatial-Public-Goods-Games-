"""Select a fixed NI strength by average update magnitude, never by cooperation."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--direction', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=96)
    parser.add_argument('--steps', type=int, default=100000)
    args = parser.parse_args()
    origin = json.loads((args.validation / 'manifest.json').read_text())
    chosen = json.loads((args.validation / 'selection.json').read_text())['selected_by_family']['learned']
    full = next(m for m in origin['models'] if m['label'] == chosen)
    models = [full] + [{'label': f'cooperation_first-{k:g}', 'method': 'cooperation_first',
                        'kappa': k, 'gate_scale': 1., 'controller': None}
                       for k in [.01, .015, .02, .025, .03, .04]]
    configs, labels = [], []
    for model, r, rho, seed in itertools.product(models, [4.4, 4.8], [0., .1], range(104, 108)):
        configs.append(Config(r=r, rho=rho, seed=seed, steps=args.steps,
                              attack='high_defect' if rho else 'none', method=model['method'],
                              kappa=model['kappa'], gate_scale=model['gate_scale'],
                              controller=tuple(model['controller']) if model['controller'] else None))
        labels.append(model['label'])
    # Reuse the earlier fixed-rule runs, with explicit source/configuration checks.
    old_manifest = json.loads((args.direction / 'manifest.json').read_text())
    old_rows = list(csv.DictReader((args.direction / 'summary.csv').open()))
    old = []
    for row in old_rows:
        if row['method'] == 'cooperation_first' and float(row['kappa']) in [.05, .1, .2]:
            name = f"run-{int(row['run']):04d}.json"
            record = json.loads((args.direction / name).read_text())
            assert record['config'] == old_manifest['configs'][int(row['run'])]
            c = record['config']
            expected = Config(r=c['r'], rho=c['rho'], seed=c['seed'], steps=100000,
                              attack='high_defect' if c['rho'] else 'none',
                              method='cooperation_first', kappa=c['kappa'])
            assert Config(**c) == expected
            old.append((name, record))
    assert len(old) == 48
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    for name in ['engine.py', 'run_pilot.py', 'run_amplitude_calibration.py',
                 'requirements.lock.txt', 'AMPLITUDE_PROTOCOL.md']:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    shutil.copyfile(args.validation / 'selection.json', source / 'model-selection.json')
    shutil.copyfile(args.direction / 'manifest.json', source / 'direction-manifest.json')
    for name, _ in old:
        shutil.copyfile(args.direction / name, source / f'direction-{name}')
    manifest = {'purpose': 'development amplitude calibration' if args.steps == 100000 else 'implementation smoke only',
                'models': models, 'configs': [asdict(c) for c in configs], 'labels': labels,
                'workers': min(args.workers, len(configs)), 'direction_path': str(args.direction.resolve()),
                'metric_columns': ['step', *METRICS],
                'sources': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    simulate(Config(L=5, steps=100, record_every=100))
    tasks = [(i, c, str(args.out)) for i, c in enumerate(configs)]
    np.random.default_rng(923).shuffle(tasks)
    results, started = [], time.perf_counter()
    with mp.get_context('spawn').Pool(manifest['workers']) as pool:
        for i, record in pool.imap_unordered(run_one, tasks):
            results.append((i, record))
            if len(results) % 16 == 0:
                print(f'{len(results)}/{len(tasks)} amplitude runs; {time.perf_counter()-started:.1f}s', flush=True)
    # Keep calibration sums independent of worker completion order.
    results.sort(key=lambda pair: pair[0])
    fields = ['run', 'label', 'method', 'kappa', 'r', 'rho', 'seed', 'steps',
              'elapsed_seconds', 'whole_cooperation', 'whole_mean_abs_ni', *METRICS]
    with (args.out / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, record in sorted(results):
            row = {key: record['config'][key] for key in fields if key in record['config']}
            row.update(record['tail'])
            row.update(run=i, label=labels[i], elapsed_seconds=record['elapsed_seconds'],
                       whole_cooperation=record['whole']['cooperation'],
                       whole_mean_abs_ni=record['whole']['mean_abs_ni'])
            writer.writerow(row)
    if args.steps == 100000:
        target = float(np.mean([record['whole']['mean_abs_ni'] for i, record in results if labels[i] == chosen]))
        assert target > 0
        records = [record for _, record in results] + [record for _, record in old]
        candidates = []
        for k in [.01, .015, .02, .025, .03, .04, .05, .1, .2]:
            runs = [rec for rec in records if rec['config']['method'] == 'cooperation_first' and rec['config']['kappa'] == k]
            assert len(runs) == 16
            amplitude = float(np.mean([rec['whole']['mean_abs_ni'] for rec in runs]))
            cells = []
            for r, rho in itertools.product([4.4, 4.8], [0., .1]):
                values = [rec['whole']['mean_abs_ni'] for rec in runs if rec['config']['r'] == r and rec['config']['rho'] == rho]
                references = [rec['whole']['mean_abs_ni'] for i, rec in results if labels[i] == chosen and rec['config']['r'] == r and rec['config']['rho'] == rho]
                cells.append({'r': r, 'rho': rho, 'fixed_mean_abs_ni': float(np.mean(values)), 'full_mean_abs_ni': float(np.mean(references))})
            candidates.append({'kappa': k, 'whole_mean_abs_ni': amplitude,
                               'distance': abs(amplitude - target), 'cells': cells})
        selected = min(candidates, key=lambda row: (row['distance'], row['kappa']))
        calibration = {'full_model': full, 'target_whole_mean_abs_ni': target,
                       'candidates': candidates, 'selected_kappa': selected['kappa'],
                       'pooled_relative_difference': selected['distance'] / target,
                       'within_10_percent_pooled': selected['distance'] / target <= .1,
                       'rule': 'minimum absolute pooled whole-trajectory NI magnitude difference; tie -> smaller kappa'}
        (args.out / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    done = {'completed_runs': len(results), 'wall_seconds': time.perf_counter() - started}
    (args.out / 'completion.json').write_text(json.dumps(done, indent=2) + '\n')
    print(done, flush=True)


if __name__ == '__main__':
    main()
