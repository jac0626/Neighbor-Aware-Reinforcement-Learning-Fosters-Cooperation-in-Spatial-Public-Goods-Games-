"""Post-training feature removal; no reoptimization or held-out seeds."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
from engine import Config, simulate
from run_pilot import run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--remove-cooperation-indicator', action='store_true',
                        help='run the separately specified w5/v5 removal check')
    args = parser.parse_args()
    original = json.loads((args.validation / 'manifest.json').read_text())
    selection = json.loads((args.validation / 'selection.json').read_text())
    label = selection['selected_by_family']['learned']
    full = next(m for m in original['models'] if m['label'] == label)
    variants = {'no_residual': [3, 8], 'no_q_proxy': [10], 'neither': [3, 8, 10]}
    if args.remove_cooperation_indicator:
        variants = {'no_cooperation_indicator': [4, 9]}
    configs, names = [], []
    for name, r, rho, seed in itertools.product(variants, [4.4, 4.8], [0., .1], range(212, 216)):
        theta = list(full['controller'])
        for index in variants[name]: theta[index] = 0.
        configs.append(Config(r=r, rho=rho, seed=seed, steps=100000,
                              attack='high_defect' if rho else 'none', method=full['method'],
                              kappa=full['kappa'], gate_scale=full['gate_scale'], controller=tuple(theta)))
        names.append(name)
    baselines = []
    for i, candidate in enumerate(original['labels']):
        if candidate == label:
            path = args.validation / f'run-{i:04d}.json'
            rec = json.loads(path.read_text())
            assert rec['config'] == original['configs'][i]
            with np.load(path.with_suffix('.npz')) as arrays:
                assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == rec['q_sha256']
                assert hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest() == rec['mask_sha256']
            baselines.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(baselines) == 16
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    for name in ['engine.py', 'run_pilot.py', 'run_feature_sensitivity.py',
                 'FEATURE_SENSITIVITY_PROTOCOL.md', 'requirements.lock.txt']:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    if args.remove_cooperation_indicator:
        shutil.copyfile(Path(__file__).parent / 'ACTION_SENSITIVITY_PROTOCOL.md', source / 'ACTION_SENSITIVITY_PROTOCOL.md')
    manifest = {'purpose': 'post-training feature sensitivity development; not retrained ablation or final test',
                'full_model': full, 'zeroed_indices': variants, 'names': names,
                'configs': [asdict(c) for c in configs], 'reused_baselines': baselines,
                'workers': min(args.workers, len(configs)),
                'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    simulate(Config(L=5, steps=100, record_every=100))
    tasks = [(i, c, str(args.out)) for i, c in enumerate(configs)]
    np.random.default_rng(925).shuffle(tasks)
    started, results = time.perf_counter(), []
    with mp.get_context('spawn').Pool(manifest['workers']) as pool:
        for i, record in pool.imap_unordered(run_one, tasks):
            results.append(i)
            if len(results) % 8 == 0:
                print(f'{len(results)}/{len(tasks)} feature-removal runs; {time.perf_counter()-started:.1f}s', flush=True)
    done = {'completed_runs': len(results), 'reused_baseline_runs': 16,
            'wall_seconds': time.perf_counter() - started}
    (args.out / 'completion.json').write_text(json.dumps(done, indent=2) + '\n')
    print(done, flush=True)


if __name__ == '__main__':
    main()
