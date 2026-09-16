"""Single-process, warmed, randomized-block timing under TIMING_PROTOCOL.md."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import numba
from engine import Config, simulate
from run_final_test import frozen_models


def main():
    parser = argparse.ArgumentParser()
    for name in ['validation', 'calibration', 'refinement', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    models = frozen_models(args.validation, args.calibration, args.refinement)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {available[0]})
    configs, labels = [], []
    rng = np.random.default_rng(927)
    for side, seed in itertools.product([30, 100], [220, 221, 222]):
        for index in rng.permutation(len(models)):
            model = models[index]
            configs.append(Config(L=side, r=4.8, rho=.1, attack='high_defect', seed=seed, steps=10000,
                                  method=model['method'], kappa=model['kappa'], gate_scale=model['gate_scale'],
                                  controller=tuple(model['controller']) if model['controller'] else None))
            labels.append(model['label'])
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'; source.mkdir()
    for name in ['engine.py', 'run_final_test.py', 'run_pilot.py', 'run_timing.py', 'TIMING_PROTOCOL.md', 'requirements.lock.txt']:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    manifest = {'purpose': 'single-process warmed execution cost; development seeds only',
                'models': models, 'configs': [asdict(c) for c in configs], 'labels': labels,
                'python': platform.python_version(), 'numpy': np.__version__, 'numba': numba.__version__,
                'initial_cpu_affinity': available, 'timing_cpu_affinity': sorted(os.sched_getaffinity(0)),
                'load_before': os.getloadavg(), 'platform': platform.platform(),
                'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}}
    (args.out / 'hardware.txt').write_text(subprocess.check_output(['lscpu'], text=True))
    (args.out / 'processes-before.txt').write_text(subprocess.check_output(['ps', '-eo', 'pid,comm,pcpu,stat', '--sort=-pcpu'], text=True))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for model in models:
        simulate(Config(L=5, steps=100, seed=100, r=4.8, rho=.1, attack='high_defect',
                        method=model['method'], kappa=model['kappa'], gate_scale=model['gate_scale'],
                        controller=tuple(model['controller']) if model['controller'] else None))
    start = time.perf_counter()
    for i, cfg in enumerate(configs):
        load = os.getloadavg()
        cpu = time.process_time(); wall = time.perf_counter()
        result = simulate(cfg)
        wall = time.perf_counter()-wall; cpu = time.process_time()-cpu
        np.savez_compressed(args.out / f'run-{i:04d}.npz', **{k: result[k] for k in ['q', 'actions', 'bad_mask', 'trajectory']})
        row = {k: result[k] for k in ['config', 'whole', 'tail']}
        row.update(label=labels[i], wall_seconds=wall, cpu_seconds=cpu, load_before=load, load_after=os.getloadavg(),
                   q_sha256=hashlib.sha256(result['q'].tobytes()).hexdigest(),
                   mask_sha256=hashlib.sha256(result['bad_mask'].tobytes()).hexdigest())
        (args.out / f'run-{i:04d}.json').write_text(json.dumps(row, indent=2) + '\n')
        print(f'{i+1}/{len(configs)} timing runs; {time.perf_counter()-start:.1f}s', flush=True)
    (args.out / 'completion.json').write_text(json.dumps({'completed_runs': len(configs), 'wall_seconds': time.perf_counter()-start,
                                                       'load_after': os.getloadavg()}, indent=2) + '\n')


if __name__ == '__main__': main()
