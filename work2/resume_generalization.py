"""Recover missing runs after the original generalization process is confirmed stopped.

Completed files and the original manifest remain unchanged. The caller must first
verify that no original process still writes to this directory.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workers', type=int, default=48)
    args = parser.parse_args()
    root = args.directory.resolve()
    if (root / 'completion.json').exists():
        parser.error('Completed batch; do not resume')
    if args.workers < 1:
        parser.error('workers must be positive')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
    # Workers import the archived engine, not a potentially edited working copy.
    sys.path.insert(0, str(root / 'source'))
    from engine import Config, METRICS, simulate
    from run_pilot import run_one
    recovery = root / f"recovery-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    recovery.mkdir()
    shutil.copyfile(__file__, recovery / 'resume_generalization.py')
    retained, missing, incomplete = [], [], []
    preserved_hashes = {}
    for i, config in enumerate(manifest['configs']):
        path = root / f'run-{i:04d}.json'
        arrays_path = path.with_suffix('.npz')
        if path.exists() and arrays_path.exists():
            record = json.loads(path.read_text())
            assert record['config'] == config
            with np.load(arrays_path) as arrays:
                assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == record['q_sha256']
                assert hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest() == record['mask_sha256']
            retained.append((i, record))
            preserved_hashes.update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, arrays_path]})
        else:
            # A one-file partial save is archived before the same run is repeated.
            for part in [path, arrays_path]:
                if part.exists():
                    shutil.move(str(part), str(recovery / part.name))
                    incomplete.append(part.name)
            missing.append((i, Config(**config), str(root)))
    plan = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'retained_runs': [i for i, _ in retained], 'rerun_ids': [i for i, _, _ in missing],
            'incomplete_files_preserved': incomplete, 'retained_file_hashes': preserved_hashes,
            'original_manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
            'workers': min(args.workers, len(missing)),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (recovery / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'{len(retained)} complete runs retained; {len(missing)} original configurations to recover', flush=True)
    started = time.perf_counter()
    if missing:
        simulate(Config(L=5, steps=100, record_every=100))
        np.random.default_rng(913).shuffle(missing)
        with mp.get_context('spawn').Pool(plan['workers']) as pool:
            for i, record in pool.imap_unordered(run_one, missing):
                retained.append((i, record))
                if len(retained) % 16 == 0 or len(retained) == len(manifest['configs']):
                    print(f'{len(retained)}/{len(manifest["configs"])} complete; recovery {time.perf_counter()-started:.1f}s', flush=True)
    assert len(retained) == len(manifest['configs'])
    for name, digest in preserved_hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    fields = ['run', 'label', 'method', 'kappa', 'L', 'M', 'state_mode', 'r', 'rho',
              'attack', 'seed', 'steps', 'elapsed_seconds', 'whole_cooperation', *METRICS]
    with (root / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, record in sorted(retained):
            row = {key: record['config'][key] for key in fields if key in record['config']}
            row.update(record['tail'])
            row.update(run=i, label=manifest['labels'][i], elapsed_seconds=record['elapsed_seconds'],
                       whole_cooperation=record['whole']['cooperation'])
            writer.writerow(row)
    done = {'completed_runs': len(retained), 'recovered': True, 'wall_seconds': None,
            'recovery_wall_seconds': time.perf_counter() - started,
            'timing_note': 'Total wall time across the interrupted process was not measured; recovery interval only.',
            'recovery_record': str(recovery.relative_to(root)), 'retained_files_unchanged': True}
    (root / 'completion.json').write_text(json.dumps(done, indent=2) + '\n')
    print(done, flush=True)


if __name__ == '__main__':
    main()
