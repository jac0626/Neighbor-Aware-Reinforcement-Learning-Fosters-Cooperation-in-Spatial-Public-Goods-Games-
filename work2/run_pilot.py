"""Development-seed factorial pilot. Never consumes the reserved final-test seeds."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import time
from dataclasses import asdict, replace
import numpy as np
import numba
from engine import Config, METRICS, simulate


def run_one(task):
    index, config, out = task
    started=time.perf_counter()
    result=simulate(config)
    elapsed=time.perf_counter()-started
    arrays={key:result.pop(key) for key in ['trajectory','actions','q','bad_mask']}
    result['elapsed_seconds']=elapsed
    result['initialization_and_action_seed']=config.seed
    result['q_sha256']=hashlib.sha256(arrays['q'].tobytes()).hexdigest()
    result['mask_sha256']=hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest()
    prefix=Path(out)/f'run-{index:04d}'
    np.savez_compressed(str(prefix)+'.npz',**arrays)
    Path(str(prefix)+'.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return index,result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--workers',type=int,default=len(os.sched_getaffinity(0)))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    # Fail rather than mix a pilot with pre-existing output.
    args.out.mkdir(parents=True,exist_ok=False)
    base=Config(steps=args.steps)
    variants=[('iql',0.),('ni_global',.5),('ni_global',1.),
              ('ni_local',.5),('ni_local',1.),('ni_local',.1),('trimmed_local',.5)]
    configs=[replace(base,r=r,rho=rho,attack='high_defect' if rho else 'none',
                     seed=seed,method=method,kappa=kappa)
             for r,rho,seed,(method,kappa) in itertools.product(
                 [3.,3.6,4.4,4.8],[0.,.1],range(100,104),variants)]
    source_dir=Path(__file__).parent
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [source_dir/'engine.py',Path(__file__)]}
    manifest={'purpose':'development pilot; not final test or published replication',
              'configs':[asdict(c) for c in configs], 'workers':min(args.workers,len(configs)),
              'sources':source_hashes,'python':platform.python_version(),
              'numpy':np.__version__,'numba':numba.__version__,
              'metric_columns':['step',*METRICS],
              'summaries':'tail: last 20 percent of executed steps; whole: all steps',
              'selected_bad':'all receivers with NI enabled, even if advantage<=0',
              'active_selected_bad':'sum of faulty selections with positive advantage / sum of positive-advantage selections',
              'timing':'worker elapsed includes simulation and metrics; excludes saving; pilot wall includes launch overhead'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Populate Numba cache once before launching independent worker processes.
    started=time.perf_counter()
    simulate(replace(base,L=5,steps=100,record_every=100))
    print(f'kernel warmup: {time.perf_counter()-started:.3f}s',flush=True)
    started=time.perf_counter()
    tasks=[(i,c,str(args.out)) for i,c in enumerate(configs)]
    np.random.default_rng(821).shuffle(tasks)
    results=[]
    with mp.get_context('spawn').Pool(manifest['workers']) as pool:
        for index,result in pool.imap_unordered(run_one,tasks):
            results.append((index,result))
            if len(results)%16==0 or len(results)==len(tasks):
                print(f'{len(results)}/{len(tasks)} runs completed; wall {time.perf_counter()-started:.1f}s',flush=True)
    with (args.out/'summary.csv').open('w') as f:
        fields=['run','method','kappa','r','rho','seed','steps','elapsed_seconds',*METRICS]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for index,result in sorted(results):
            row={key:result['config'][key] for key in fields if key in result['config']}
            row.update(result['tail']);row.update(run=index,elapsed_seconds=result['elapsed_seconds'])
            writer.writerow(row)
    completion={'completed_runs':len(results),'wall_seconds':time.perf_counter()-started}
    (args.out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion),flush=True)

if __name__=='__main__': main()
