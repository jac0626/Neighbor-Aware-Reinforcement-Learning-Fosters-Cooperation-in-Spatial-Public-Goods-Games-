"""Export validation tables and unchanged computed plots to the thesis draft."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--thesis',type=Path,default=Path('thesis'))
    parser.add_argument('--tag',default='development',help='Separate output namespace for each validation batch')
    args=parser.parse_args()
    root=args.directory;dest=args.thesis/'figures'/f'work2-{args.tag}';dest.mkdir(parents=True,exist_ok=True)
    selection=json.loads((root/'selection.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    rows=list(csv.DictReader((root/'summary.csv').open()))
    groups=defaultdict(list)
    for row in rows:groups[(row['label'],float(row['r']),float(row['rho']))].append(row)
    method_names={'iql':'独立Q-learning','ni_global':'全局NI','ni_local':'局部NI',
                  'trimmed_local':'上尾过滤','cooperation_first':'合作优先','cooperation_only':'仅合作方向'}
    labels=[(m['label'],method_names[m['method']]+(f'，$\\kappa={m["kappa"]:g}$' if m['kappa'] else ''))
            for m in manifest['models'] if m['controller'] is None]
    labels.append((selection['selected_candidate'],'所选学习控制器'))
    suffix='' if args.tag=='development' else '-'+args.tag
    repeats=len({row['seed'] for row in rows})
    lines=[r'\begin{table}[!htbp]',r'\centering\small',
           rf'\caption{{独立验证的末段合作率：{repeats}次运行的均值$\pm$样本标准差，数值保留三位小数}}',
           rf'\label{{tab:w2-validation{suffix}}}',r'\begin{tabular}{lcccc}',r'\toprule',
           r'方法 & $r=4.4$干净 & $r=4.4$受扰 & $r=4.8$干净 & $r=4.8$受扰 \\',r'\midrule']
    cells=[(4.4,0.),(4.4,.1),(4.8,0.),(4.8,.1)]
    for label,name in labels:
        values=[]
        for r,rho in cells:
            cell=np.array([float(row['cooperation']) for row in groups[(label,r,rho)]])
            values.append(f'${cell.mean():.3f}\\pm{cell.std(ddof=1):.3f}$')
        lines.append(name+' & '+' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (dest/'validation-table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\caption{各独立训练候选在验证集上的全程合作目标}',
           rf'\label{{tab:w2-training-validation{suffix}}}',r'\begin{tabular}{lcc}',r'\toprule',
           r'优化初始化 & 最终分布均值参数 & 末代最佳候选参数 \\',r'\midrule']
    optimizers=sorted({m['label'].split('-')[0] for m in manifest['models'] if m['controller'] is not None})
    for opt in optimizers:
        scores=selection['scores'];lines.append(f'{opt} & {scores[str(opt)+"-final_mean"]:.4f} & {scores[str(opt)+"-last_generation_best"]:.4f}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (dest/'training-validation-table.tex').write_text('\n'.join(lines)+'\n')
    for name in ['trajectories.pdf','snapshots.pdf']:shutil.copyfile(root/name,dest/name)
    provenance={'source_directory':str(root.resolve()),'purpose':'validation/model-selection data, not final test',
                'selected_candidate':selection['selected_candidate'],
                'source_files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                for name in ['manifest.json','selection.json','summary.csv','trajectories.pdf','snapshots.pdf']},
                'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    provenance['exported_files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir()}
    (args.thesis/'validation'/f'work2-{args.tag}-sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Exported two validation tables and two figures')

if __name__=='__main__':main()
