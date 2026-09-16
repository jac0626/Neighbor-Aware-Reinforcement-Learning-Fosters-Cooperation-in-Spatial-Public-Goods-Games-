"""Show every optimizer repetition in the common development validation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
parser.add_argument('--thesis', type=Path, required=True)
args = parser.parse_args()
root = args.directory
manifest = json.loads((root / 'manifest.json').read_text())
selection = json.loads((root / 'selection.json').read_text())
records = list(csv.DictReader((root / 'summary.csv').open()))
assert len(records) == len(manifest['configs']) == 512
families = {'learned': '完整模型', 'cooperation_gate': '固定评分门控',
            'selection_only': '仅学习评分', 'gate_only': '最高奖励门控'}
tex = [r'\begin{table}[!htbp]', r'\centering\small',
       r'\caption{共同验证中全部训练输出的全程目标；每类三个优化初始化}',
       r'\label{tab:w2-training-repeats}', r'\begin{tabular}{lcrr}', r'\toprule',
       r'结构 & 优化初始化 & 最终分布均值 & 末代最佳候选 \\', r'\midrule']
values = {}
for family, name in families.items():
    models = [m for m in manifest['models'] if m.get('family') == family]
    assert len(models) == 6
    seeds = sorted({int(m['label'].split('-')[0]) for m in models})
    assert len(seeds) == 3
    for seed in seeds:
        cells = []
        for suffix in ['final_mean', 'last_generation_best']:
            label = f'{seed}-{suffix}'
            rows = [row for row in records if row['label'] == label]
            assert len(rows) == 16
            score = float(np.mean([float(row['whole_cooperation']) for row in rows]))
            assert np.isclose(score, selection['scores'][label], rtol=0, atol=1e-12)
            values[label] = score
            cells.append(f'{score:.6f}' + (r'$^*$' if selection['selected_by_family'][family] == label else ''))
        tex.append(f'{name} & {seed} & ' + ' & '.join(cells) + r' \\')
tex += [r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\footnotesize 星号为该类选定输出；同一初始化的两列不是独立训练。', r'\end{table}']
target = args.thesis / 'figures/work2-training-repeats.tex'
target.write_text('\n'.join(tex) + '\n')
provenance = {'scores': values, 'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ['manifest.json', 'selection.json', 'summary.csv']},
              'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(args.thesis / 'validation/work2-training-repeats.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps(values, indent=2))
