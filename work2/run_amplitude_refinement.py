"""One recorded interpolation step using NI magnitude only."""
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
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = json.loads((args.calibration / 'calibration.json').read_text())
    audit = json.loads((args.calibration / 'verification.json').read_text())
    assert audit['sources']['calibration.json'] == hashlib.sha256((args.calibration / 'calibration.json').read_bytes()).hexdigest()
    left, right = [next(x for x in base['candidates'] if x['kappa'] == k) for k in [.025, .03]]
    target = base['target_whole_mean_abs_ni']
    assert left['whole_mean_abs_ni'] < target < right['whole_mean_abs_ni']
    kappa = left['kappa'] + (target-left['whole_mean_abs_ni'])/(right['whole_mean_abs_ni']-left['whole_mean_abs_ni'])*(right['kappa']-left['kappa'])
    configs = [Config(r=r, rho=rho, seed=seed, steps=100000, method='cooperation_first', kappa=kappa,
                      attack='high_defect' if rho else 'none')
               for r, rho, seed in itertools.product([4.4, 4.8], [0., .1], range(104, 108))]
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'; source.mkdir()
    for name in ['engine.py', 'run_pilot.py', 'run_amplitude_refinement.py', 'AMPLITUDE_REFINEMENT_PROTOCOL.md', 'requirements.lock.txt']:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    shutil.copyfile(args.calibration / 'calibration.json', source / 'base-calibration.json')
    manifest = {'purpose': 'one adaptive interpolation for amplitude calibration; not final test',
                'base_path': str(args.calibration.resolve()), 'interpolated_kappa': kappa,
                'configs': [asdict(c) for c in configs], 'workers': 16,
                'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Frozen interpolation kappa={kappa:.17g}; 16 original development configurations', flush=True)
    simulate(Config(L=5, steps=100, record_every=100))
    results, started = [], time.perf_counter()
    tasks = [(i, c, str(args.out)) for i, c in enumerate(configs)]
    np.random.default_rng(926).shuffle(tasks)
    with mp.get_context('spawn').Pool(16) as pool:
        for i, rec in pool.imap_unordered(run_one, tasks):
            results.append((i, rec))
            print(f'{len(results)}/16 refinement runs; {time.perf_counter()-started:.1f}s', flush=True)
    results.sort(key=lambda pair: pair[0])
    value = float(np.mean([rec['whole']['mean_abs_ni'] for _, rec in results]))
    candidates = [{'kappa': x['kappa'], 'whole_mean_abs_ni': x['whole_mean_abs_ni']} for x in base['candidates']]
    candidates.append({'kappa': kappa, 'whole_mean_abs_ni': value})
    chosen = min(candidates, key=lambda x: (abs(x['whole_mean_abs_ni']-target), x['kappa']))
    result = {'target_whole_mean_abs_ni': target, 'candidates': candidates,
              'selected_kappa': chosen['kappa'], 'selected_whole_mean_abs_ni': chosen['whole_mean_abs_ni'],
              'pooled_relative_difference': abs(chosen['whole_mean_abs_ni']-target)/target,
              'within_10_percent_pooled': abs(chosen['whole_mean_abs_ni']-target)/target <= .1,
              'base_calibration_sha256': manifest['sources']['base-calibration.json']}
    (args.out / 'refinement.json').write_text(json.dumps(result, indent=2) + '\n')
    done = {'completed_runs': len(results), 'wall_seconds': time.perf_counter()-started}
    (args.out / 'completion.json').write_text(json.dumps(done, indent=2) + '\n'); print(done, flush=True)


if __name__ == '__main__': main()
