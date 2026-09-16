"""Development comparison of direction constraints and fixed NI strengths."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
import argparse,csv,hashlib,itertools,json,multiprocessing as mp,shutil,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from engine import Config,METRICS,simulate
from run_pilot import run_one


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=96);p.add_argument('--steps',type=int,default=100000)
    args=p.parse_args()
    models=[]
    for method,k in [('iql',0.),('ni_global',.5),('trimmed_local',.5)]:
        models.append({'label':f'{method}-{k:g}','method':method,'kappa':k,'controller':None})
    for method,k in itertools.product(['cooperation_first','cooperation_only'],[.05,.1,.2,.5]):
        models.append({'label':f'{method}-{k:g}','method':method,'kappa':k,'controller':None})
    models.append({'label':'semantic-prior','method':'learned','kappa':.5,
                   'controller':[1.,0.,0.,0.,2.,0.,0.,0.,0.,0.,0.,8.]})
    configs=[];labels=[]
    for model,r,rho,seed in itertools.product(models,[4.4,4.8],[0.,.1],range(104,108)):
        configs.append(Config(r=r,steps=args.steps,seed=seed,rho=rho,attack='high_defect' if rho else 'none',
                       method=model['method'],kappa=model['kappa'],
                       controller=tuple(model['controller']) if model['controller'] else None))
        labels.append(model['label'])
    args.out.mkdir(parents=True,exist_ok=False);source=args.out/'source';source.mkdir()
    for name in ['engine.py','run_pilot.py','run_direction_pilot.py','requirements.lock.txt']:
        shutil.copyfile(Path(__file__).parent/name,source/name)
    manifest={'purpose':'direction/strength development; untrained semantic prior, not final test',
              'models':models,'labels':labels,'configs':[asdict(c) for c in configs],
              'metric_columns':['step',*METRICS],'workers':args.workers,
              'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},
              'selection_rule':'compare equal-weight whole-trajectory cooperation over four cells; fixed strength chosen only on development seeds'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    simulate(Config(L=5,steps=100,record_every=100))
    tasks=[(i,c,str(args.out)) for i,c in enumerate(configs)];np.random.default_rng(921).shuffle(tasks)
    results=[];started=time.perf_counter()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for i,result in pool.imap_unordered(run_one,tasks):
            results.append((i,result))
            if len(results)%16==0:print(f'{len(results)}/{len(tasks)} direction runs; {time.perf_counter()-started:.1f}s',flush=True)
    fields=['run','label','method','kappa','r','rho','seed','steps','elapsed_seconds','whole_cooperation',*METRICS]
    with (args.out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,result in sorted(results):
            row={k:result['config'][k] for k in fields if k in result['config']}
            row.update(result['tail']);row.update(run=i,label=labels[i],elapsed_seconds=result['elapsed_seconds'],
                                                 whole_cooperation=result['whole']['cooperation']);writer.writerow(row)
    scores={m['label']:float(np.mean([v['whole']['cooperation'] for i,v in results if labels[i]==m['label']])) for m in models}
    selection={'scores':scores,'selected_candidate':'semantic-prior','selected_model':models[-1],
               'note':'Untrained analytic initialization; no learned candidate selection in this pilot'}
    (args.out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    done={'completed_runs':len(results),'wall_seconds':time.perf_counter()-started}
    (args.out/'completion.json').write_text(json.dumps(done,indent=2)+'\n');print(done,flush=True)

if __name__=='__main__':main()
