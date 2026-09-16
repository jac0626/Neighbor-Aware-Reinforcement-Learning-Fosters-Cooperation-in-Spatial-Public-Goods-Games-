"""Freeze the selected models and execute FINAL_TEST_PROTOCOL.md once."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import numba
from engine import Config, METRICS, simulate
from run_pilot import run_one


def frozen_models(validation, calibration, refinement):
    origin = json.loads((validation / 'manifest.json').read_text())
    selection = json.loads((validation / 'selection.json').read_text())
    done = json.loads((validation / 'completion.json').read_text())
    assert done['completed_runs'] == len(origin['configs'])
    model_labels = [selection['selected_by_family'][family] for family in
                    ['learned', 'cooperation_gate', 'selection_only', 'gate_only']]
    models = [next(m for m in origin['models'] if m['label'] == label) for label in model_labels]
    calibrated = json.loads((calibration / 'calibration.json').read_text())
    calibration_audit = json.loads((calibration / 'verification.json').read_text())
    assert calibration_audit['verified_new_runs'] == 112
    assert calibration_audit['verified_reused_runs'] == 48
    assert calibration_audit['selected_kappa'] == calibrated['selected_kappa']
    assert calibration_audit['sources']['calibration.json'] == hashlib.sha256((calibration / 'calibration.json').read_bytes()).hexdigest()
    assert calibrated['full_model'] == models[0]
    cal_manifest = json.loads((calibration / 'manifest.json').read_text())
    cal_done = json.loads((calibration / 'completion.json').read_text())
    assert cal_done['completed_runs'] == len(cal_manifest['configs']) == 112
    assert all(c['steps'] == 100000 for c in cal_manifest['configs'])
    refined = json.loads((refinement / 'refinement.json').read_text())
    refinement_audit = json.loads((refinement / 'verification.json').read_text())
    assert refinement_audit['verified_new_runs'] == refinement_audit['verified_reused_runs'] == 16
    assert refinement_audit['sources']['refinement.json'] == hashlib.sha256((refinement / 'refinement.json').read_bytes()).hexdigest()
    assert refined['base_calibration_sha256'] == hashlib.sha256((calibration / 'calibration.json').read_bytes()).hexdigest()
    assert refined['selected_kappa'] == refinement_audit['selected_kappa']
    assert refined['within_10_percent_pooled']
    variants = [('iql', 0.), ('ni_global', .5), ('ni_local', .5),
                ('trimmed_local', .5), ('cooperation_first', .2),
                ('cooperation_first', refined['selected_kappa'])]
    for method, kappa in dict.fromkeys(variants):
        models.append({'label': f'{method}-{kappa:g}', 'method': method,
                       'kappa': kappa, 'gate_scale': 1., 'controller': None})
    assert len({m['label'] for m in models}) == len(models)
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--refinement', type=Path, required=True)
    parser.add_argument('--generalization', type=Path, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=96)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    models = frozen_models(args.validation, args.calibration, args.refinement)
    engine_hash = hashlib.sha256((Path(__file__).parent / 'engine.py').read_bytes()).hexdigest()
    generalization_records = []
    for path in args.generalization:
        manifest = json.loads((path / 'manifest.json').read_text())
        done = json.loads((path / 'completion.json').read_text())
        audit = json.loads((path / 'generalization.json').read_text())
        assert done['completed_runs'] == audit['verified_runs'] == len(manifest['configs'])
        expected_runs = {'messages': 384, 'parameters': 336, 'state': 288, 'scale': 144}
        assert done['completed_runs'] == expected_runs[manifest['suite']]
        assert {c['seed'] for c in manifest['configs']} == {216, 217, 218, 219}
        assert all(c['steps'] == 100000 and 216 <= c['seed'] <= 219 for c in manifest['configs'])
        assert manifest['sources']['engine.py'] == engine_hash
        generalization_records.append({'path': str(path.resolve()), 'suite': manifest['suite'],
                                       'audit_sha256': hashlib.sha256((path / 'generalization.json').read_bytes()).hexdigest()})
    assert {item['suite'] for item in generalization_records} == {'messages', 'parameters', 'state', 'scale'}
    configs, labels = [], []
    for model, r, rho, seed in itertools.product(models, [4.4, 4.8], [0., .1], range(1000, 1030)):
        configs.append(Config(r=r, rho=rho, seed=seed, steps=100000,
                              attack='high_defect' if rho else 'none', method=model['method'],
                              kappa=model['kappa'], gate_scale=model['gate_scale'],
                              controller=tuple(model['controller']) if model['controller'] else None))
        labels.append(model['label'])
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    for name in ['engine.py', 'run_pilot.py', 'run_final_test.py', 'test_engine.py',
                 'requirements.lock.txt', 'FINAL_TEST_PROTOCOL.md',
                 'paired_analysis.py', 'summarize_final_test.py',
                 'AMPLITUDE_PROTOCOL.md', 'AMPLITUDE_REFINEMENT_PROTOCOL.md',
                 'FINAL_FIGURE_PLAN.md', 'export_final_tex.py']:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    shutil.copyfile(args.validation / 'selection.json', source / 'model-selection.json')
    shutil.copyfile(args.calibration / 'calibration.json', source / 'amplitude-calibration.json')
    shutil.copyfile(args.refinement / 'refinement.json', source / 'amplitude-refinement.json')
    manifest = {'purpose': 'final held-out test of frozen selected controllers',
                'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'models': models, 'configs': [asdict(c) for c in configs], 'labels': labels,
                'workers': min(args.workers, len(configs)), 'metric_columns': ['step', *METRICS],
                'python': platform.python_version(), 'numpy': np.__version__, 'numba': numba.__version__,
                'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'primary_comparisons': [{'reference': models[0]['label'], 'baseline': baseline}
                                        for baseline in ['cooperation_first-0.2', models[1]['label']]],
                'generalization_development': generalization_records,
                'sources': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # A separate development seed warms the kernel; no held-out run is discarded.
    simulate(Config(L=5, steps=100, record_every=100, seed=100))
    tasks = [(i, c, str(args.out)) for i, c in enumerate(configs)]
    np.random.default_rng(924).shuffle(tasks)
    results, started = [], time.perf_counter()
    with mp.get_context('spawn').Pool(manifest['workers']) as pool:
        for i, record in pool.imap_unordered(run_one, tasks):
            results.append((i, record))
            if len(results) % 16 == 0 or len(results) == len(tasks):
                print(f'{len(results)}/{len(tasks)} held-out runs; {time.perf_counter()-started:.1f}s', flush=True)
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
    done = {'completed_runs': len(results), 'wall_seconds': time.perf_counter() - started}
    (args.out / 'completion.json').write_text(json.dumps(done, indent=2) + '\n')
    print(done, flush=True)


if __name__ == '__main__':
    main()
