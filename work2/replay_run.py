"""Replay one recorded evaluation using that batch's archived engine."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path, help='completed evaluation batch with manifest.json')
    parser.add_argument('--run', type=int, required=True, help='zero-based recorded run index')
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if not 0 <= args.run < len(manifest['configs']):
        parser.error('run is outside this manifest')
    engine = root / 'source/engine.py'
    assert hashlib.sha256(engine.read_bytes()).hexdigest() == manifest['sources']['engine.py']
    original_path = root / f'run-{args.run:04d}.json'
    original = json.loads(original_path.read_text())
    config = manifest['configs'][args.run]
    assert original['config'] == config
    sys.path.insert(0, str(engine.parent))
    from engine import Config, simulate
    args.out.mkdir(parents=True, exist_ok=False)
    result = simulate(Config(**config))
    np.savez_compressed(args.out / 'replayed.npz', **{key: result[key] for key in ['q', 'actions', 'bad_mask', 'trajectory']})
    with np.load(original_path.with_suffix('.npz')) as arrays:
        checks = {key: bool(np.array_equal(result[key], arrays[key], equal_nan=True))
                  for key in ['q', 'actions', 'bad_mask', 'trajectory']}
    checks.update({key: result[key] == original[key] for key in ['whole', 'tail']})
    report = {'original_directory': str(root), 'run': args.run, 'config': config,
              'archived_engine_sha256': manifest['sources']['engine.py'],
              'checks': checks, 'exact_match': all(checks.values()),
              'whole': result['whole'], 'tail': result['tail']}
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'run': args.run, 'exact_match': report['exact_match'], 'checks': checks}))


if __name__ == '__main__': main()
