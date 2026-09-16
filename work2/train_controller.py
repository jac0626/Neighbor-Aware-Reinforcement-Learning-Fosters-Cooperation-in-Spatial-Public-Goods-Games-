"""Seeded CEM on full online-learning trajectories, with archived training episodes."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
from engine import Config, simulate


def evaluate(task):
    candidate,cell,config=task
    start=time.perf_counter();result=simulate(config)
    return {'candidate':candidate,'cell':cell,'config':asdict(config),
            'objective':result['whole']['raw_welfare_per_agent']/(5*(config.r-1)),
            'whole':result['whole'],'tail':result['tail'],
            'q_sha256':hashlib.sha256(result['q'].tobytes()).hexdigest(),
            'seconds':time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--optimizer-seeds',type=int,nargs='+',default=[3000,3001,3002])
    p.add_argument('--population',type=int,default=24)
    p.add_argument('--generations',type=int,default=12)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--workers',type=int,default=len(os.sched_getaffinity(0)))
    p.add_argument('--method',choices=['learned','gate_only','selection_only','cooperation_gate'],default='learned')
    args=p.parse_args()
    if args.population<5 or args.generations<1 or args.workers<1: p.error('Invalid optimization budget')
    args.out.mkdir(parents=True,exist_ok=False)
    source=args.out/'source';source.mkdir()
    for name in ['engine.py','train_controller.py','requirements.lock.txt']:
        shutil.copyfile(Path(__file__).parent/name,source/name)
    manifest={**vars(args),'out':str(args.out),'objective':'equal-weight mean whole-trajectory normalized true welfare across four r/fault cells',
              'train_seed_pool':[110,199],'validation_seeds_used':[], 'test_seeds_used':[],
              'feature_names':['reward_advantage','action_agreement','median_deviation','own_experience_residual','reported_cooperation'],
              'initial_controller':[1.,0.,0.,0.,2.,0.,0.,0.,0.,0.,0.,float(np.log(2./3.))],
              'initial_gate':.4, 'max_kappa':.5,
              'selection_only_fixed_gate':.4,
              'elite_fraction':.2,'std_floor':.15,'update_rate':.8,'parameter_bounds':[-8,8],
              'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    initial=np.array(manifest["initial_controller"])
    # This training ablation fixes the analytic cooperation-priority score and
    # learns only the gate; execution uses the unchanged learned controller.
    simulation_method='learned' if args.method=='cooperation_gate' else args.method
    simulate(Config(L=5,steps=100,record_every=100,method=simulation_method,controller=tuple(initial)))
    start=time.perf_counter()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for optimizer_seed in args.optimizer_seeds:
            folder=args.out/f'optimizer-{optimizer_seed}';folder.mkdir()
            rng=np.random.default_rng(optimizer_seed)
            mean=initial.copy();std=np.array([1.]*5+[2.]*7)
            if args.method in ('gate_only','cooperation_gate'): active=np.arange(5,12)
            elif args.method=='selection_only': active=np.arange(5)
            else: active=np.arange(12)
            for generation in range(args.generations):
                candidates=np.tile(mean,(args.population,1))
                candidates[:,active]=np.clip(rng.normal(mean[active],std[active],(args.population,len(active))),-8,8)
                candidates[0]=mean
                seed=int(rng.integers(110,200))
                tasks=[]
                for index,theta in enumerate(candidates):
                    for cell,(r,rho) in enumerate([(4.4,0.),(4.4,.1),(4.8,0.),(4.8,.1)]):
                        cfg=Config(r=r,steps=args.steps,seed=seed,method=simulation_method,kappa=.5,
                                   controller=tuple(theta),gate_scale=.4 if args.method=='selection_only' else 1.,
                                   rho=rho,attack='high_defect' if rho else 'none')
                        tasks.append((index,cell,cfg))
                gen_start=time.perf_counter()
                records=list(pool.imap_unordered(evaluate,tasks))
                records.sort(key=lambda row:(row['candidate'],row['cell']))
                scores=np.zeros(args.population)
                for record in records: scores[record['candidate']]+=record['objective']/4
                elite_count=max(2,int(np.ceil(.2*args.population)))
                elite_indices=np.argsort(-scores,kind='stable')[:elite_count]
                elites=candidates[elite_indices]
                mean[active]=.2*mean[active]+.8*elites[:,active].mean(axis=0)
                std[active]=np.maximum(.15,.2*std[active]+.8*elites[:,active].std(axis=0))
                best=candidates[elite_indices[0]]
                record={'generation':generation,'environment_seed':seed,'candidates':candidates.tolist(),
                        'scores':scores.tolist(),'elites':elite_indices.tolist(),'mean':mean.tolist(),'std':std.tolist(),
                        'best':best.tolist(),'episodes':records,'elapsed_seconds':time.perf_counter()-gen_start}
                (folder/f'generation-{generation:03d}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
                print(f'optimizer={optimizer_seed} generation={generation+1}/{args.generations} '
                      f'best={scores.max():.4f} population_mean={scores.mean():.4f} '
                      f'seconds={record["elapsed_seconds"]:.1f}',flush=True)
            (folder/'candidates.json').write_text(json.dumps({'optimizer_seed':optimizer_seed,'method':simulation_method,
                'training_variant':args.method,
                'kappa':.5,'gate_scale':.4 if args.method=='selection_only' else 1.,
                'final_mean':mean.tolist(),'last_generation_best':best.tolist()},indent=2)+'\n')
    result={'optimizers_completed':args.optimizer_seeds,'episodes':len(args.optimizer_seeds)*args.generations*args.population*4,
            'wall_seconds':time.perf_counter()-start}
    (args.out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
