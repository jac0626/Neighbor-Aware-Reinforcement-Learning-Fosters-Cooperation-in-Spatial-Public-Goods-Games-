"""Independent validation of frozen candidate vectors and simple baselines."""
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
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--compare-training',type=Path,nargs='*',default=[],help='Additional completed training batches for matched ablation validation')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--seeds',type=int,nargs='+',default=[200,201,202,203])
    p.add_argument('--workers',type=int,default=96);args=p.parse_args()
    if not all(200<=s<=219 for s in args.seeds):p.error('This script is validation-only; seeds must be 200..219')
    models=[]
    for training in [args.training,*args.compare_training]:
        done=json.loads((training/'completion.json').read_text())
        for opt_seed in done['optimizers_completed']:
            path=training/f'optimizer-{opt_seed}'/'candidates.json'
            record=json.loads(path.read_text())
            for key in ['final_mean','last_generation_best']:
                models.append({'label':f'{opt_seed}-{key}','method':record['method'],'kappa':record['kappa'],'gate_scale':record['gate_scale'],
                               'family':record.get('training_variant',record['method']),
                               'training_path':str(training.resolve()),
                               'controller':record[key],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if len({m['label'] for m in models})!=len(models):
        p.error('Compared training batches must have distinct optimizer seeds')
    for method,k in [('iql',0.),('ni_global',.5),('ni_local',.1),('ni_local',.5),('trimmed_local',.5),('cooperation_first',.5),('cooperation_first',.2),('cooperation_only',.2)]:
        models.append({'label':f'{method}-{k:g}','method':method,'kappa':k,'gate_scale':1.,'controller':None})
    args.out.mkdir(parents=True,exist_ok=False);source=args.out/'source';source.mkdir()
    for name in ['engine.py','run_pilot.py','evaluate_controllers.py','requirements.lock.txt']:
        shutil.copyfile(Path(__file__).parent/name,source/name)
    configs=[];labels=[]
    for model,r,rho,seed in itertools.product(models,[4.4,4.8],[0.,.1],args.seeds):
        cfg=Config(r=r,steps=args.steps,seed=seed,method=model['method'],kappa=model['kappa'],
                   controller=tuple(model['controller']) if model['controller'] else None,gate_scale=model['gate_scale'],
                   rho=rho,attack='high_defect' if rho else 'none')
        configs.append(cfg);labels.append(model['label'])
    manifest={'purpose':'independent validation, not final test','models':models,'labels':labels,
              'configs':[asdict(c) for c in configs],'workers':args.workers,'metric_columns':['step',*METRICS],
              'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},
              'selection_rule':'equal-weight mean whole-trajectory cooperation over four r/fault cells; report tail separately',
              'training_paths':[str(t.resolve()) for t in [args.training,*args.compare_training]]}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    simulate(Config(L=5,steps=100,record_every=100))
    tasks=[(i,c,str(args.out)) for i,c in enumerate(configs)]
    np.random.default_rng(912).shuffle(tasks);results=[];start=time.perf_counter()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for i,result in pool.imap_unordered(run_one,tasks):
            results.append((i,result))
            if len(results)%16==0:print(f'{len(results)}/{len(tasks)} validation runs; {time.perf_counter()-start:.1f}s',flush=True)
    fields=['run','label','method','kappa','r','rho','seed','steps','elapsed_seconds','whole_cooperation',*METRICS]
    with (args.out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,result in sorted(results):
            row={k:result['config'][k] for k in fields if k in result['config']}
            row.update(result['tail']);row.update(run=i,label=labels[i],elapsed_seconds=result['elapsed_seconds'],
                                                 whole_cooperation=result['whole']['cooperation'])
            writer.writerow(row)
    scores={label:np.mean([result['whole']['cooperation'] for i,result in results if labels[i]==label]) for label in labels}
    candidate_scores={m['label']:scores[m['label']] for m in models if m['controller'] is not None}
    selected=max(candidate_scores,key=candidate_scores.get)
    selected_by_family={}
    for model in models:
        if model['controller'] is None:continue
        family=model['family'];label=model['label']
        if family not in selected_by_family or scores[label]>scores[selected_by_family[family]]:
            selected_by_family[family]=label
    (args.out/'selection.json').write_text(json.dumps({'scores':scores,'selected_candidate':selected,
        'selected_by_family':selected_by_family,
        'selected_model':next(m for m in models if m['label']==selected)},indent=2)+'\n')
    completion={'completed_runs':len(results),'wall_seconds':time.perf_counter()-start}
    (args.out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n');print(completion,flush=True)

if __name__=='__main__':main()
