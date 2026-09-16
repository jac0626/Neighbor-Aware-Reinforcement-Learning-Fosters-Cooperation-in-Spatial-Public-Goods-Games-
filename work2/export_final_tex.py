"""Export an audited final test, preserving all methods and independent seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--thesis', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / 'manifest.json').read_text())
    result = json.loads((root / 'final-report.json').read_text())
    done = json.loads((root / 'completion.json').read_text())
    assert result['verified_runs'] == done['completed_runs'] == len(manifest['configs']) == 1200
    for name, digest in result['sources'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    labels = [m['label'] for m in manifest['models']]
    names = ['完整模型', '固定评分/学习门控', '学习评分/固定门控', '最高奖励/学习门控',
             '独立Q-learning', '全局NI', '局部NI', '上尾过滤', '固定合作规则', '幅度校准固定规则']
    assert len(labels) == len(names) == 10
    lookup = {(x['label'], x['r'], x['rho']): x for x in result['descriptive']}
    assert len(lookup) == 40
    for x in lookup.values():
        assert x['seeds'] == list(range(1000, 1030))
        assert len(x['whole']) == len(x['tail']) == 30
    dst = args.thesis / 'figures/work2-final'
    dst.mkdir(parents=True, exist_ok=True)
    for stem, caption in [('whole', '最终测试的全程合作均值与运行间标准差'),
                          ('tail', '最终测试的末段合作均值与运行间标准差')]:
        tex = [r'\begin{table}[!htbp]', r'\centering\small', r'\caption{' + caption + '}',
               r'\label{tab:w2-final-' + stem + '}', r'\begin{tabular}{clrr}', r'\toprule',
               r'$r$ & 方法 & 干净 & 10\%持续失真 \\', r'\midrule']
        for r in [4.4, 4.8]:
            for label, name in zip(labels, names):
                cells = []
                for rho in [0., .1]:
                    values = np.array(lookup[label, r, rho][stem])
                    cells.append(f'${values.mean():.5f}\\pm{values.std(ddof=1):.5f}$')
                tex.append(f'{r:g} & {name} & ' + ' & '.join(cells) + r' \\')
            if r == 4.4: tex.append(r'\midrule')
        tex += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip\footnotesize 数据来源：本文冻结后的留出测试；每格30次独立运行。', r'\end{table}']
        (dst / f'{stem}-table.tex').write_text('\n'.join(tex) + '\n')
    for name in ['primary-table.tex', 'primary-differences.pdf', 'primary-differences.png',
                 'distributions-whole.pdf', 'distributions-whole.png',
                 'distributions-tail.pdf', 'distributions-tail.png', 'trajectories.pdf', 'trajectories.png']:
        shutil.copyfile(root / name, dst / name)
    # The smallest held-out seed is selected before results are available.
    snapshot_seed = 1000
    indices = [0, 1, 5, 8]
    panel_names = ['Full controller', 'Fixed C score + gate', 'Global NI', 'Fixed C rule']
    snapshots = []
    for r in [4.4, 4.8]:
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), layout='constrained')
        for ax, index, name in zip(axes.flat, indices, panel_names):
            candidates = [i for i, (label, config) in enumerate(zip(manifest['labels'], manifest['configs']))
                          if label == labels[index] and config['r'] == r and config['rho'] == .1 and config['seed'] == snapshot_seed]
            assert len(candidates) == 1
            i = candidates[0]
            rec = json.loads((root / f'run-{i:04d}.json').read_text())
            with np.load(root / f'run-{i:04d}.npz') as arrays:
                assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == rec['q_sha256']
                actions = arrays['actions'].reshape(30, 30)
                snapshot_cooperation = float((actions == 0).mean())
            snapshots.append({'r': r, 'rho': .1, 'label': labels[index], 'seed': snapshot_seed,
                              'run': i, 'final_action_cooperation': snapshot_cooperation})
            ax.imshow(actions, cmap=ListedColormap(['#3575a8', '#d2d2d2']), vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'{name}\nFinal-step cooperation={snapshot_cooperation:.3f}', fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'r={r:g}, 10% high-D messages, fixed example seed={snapshot_seed}', fontsize=11)
        fig.savefig(dst / f'snapshots-r{r:g}.pdf'); fig.savefig(dst / f'snapshots-r{r:g}.png', dpi=180)
        plt.close(fig)
    export = {'verified_runs': 1200, 'source_path': str(root.resolve()),
              'source_sha256': hashlib.sha256((root / 'final-report.json').read_bytes()).hexdigest(),
              'snapshot_selection': 'smallest held-out seed 1000, fixed before test', 'snapshots': snapshots,
              'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir()}}
    (args.thesis / 'validation/work2-final-sources.json').write_text(json.dumps(export, indent=2) + '\n')
    (root / 'export-analysis-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'exported_methods': len(labels), 'verified_runs': 1200, 'snapshot_seed': snapshot_seed}))


if __name__ == '__main__': main()
