"""Describe prespecified secondary metrics after the full held-out audit.

This adds no statistical tests or model selection. All conditions and methods
remain available in final-report.json; the table explains the two fault cells.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
parser.add_argument('--thesis', type=Path, required=True)
args = parser.parse_args()
root = args.directory
report = json.loads((root / 'final-report.json').read_text())
manifest = json.loads((root / 'manifest.json').read_text())
assert report['verified_runs'] == len(manifest['configs']) == 1200
for name, digest in report['sources'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
labels = [m['label'] for m in manifest['models']]
lookup = {(x['label'], x['r'], x['rho']): x for x in report['descriptive']}
names = {0: '完整模型', 1: '固定评分门控', 5: '全局NI', 8: '固定合作规则', 9: '幅度校准固定规则'}
metrics = ['selected_bad_fraction', 'mean_gate', 'active_ni_fraction', 'mean_abs_ni']
rows = []
tex = [r'\begin{table}[!htbp]', r'\centering\small',
       r'\caption{最终测试受扰条件下的信息使用；各项为30次全程指标的均值}',
       r'\label{tab:w2-final-mechanism}', r'\begin{tabular}{clrrrr}', r'\toprule',
       r'$r$ & 方法 & 异常来源/\% & 门控均值 & NI触发/\% & 绝对NI \\', r'\midrule']
for r in [4.4, 4.8]:
    for index, name in names.items():
        item = lookup[labels[index], r, .1]
        assert item['seeds'] == list(range(1000, 1030))
        values = {metric: np.asarray(item['whole_metrics'][metric], dtype=float) for metric in metrics}
        assert all(v.shape == (30,) and np.isfinite(v).all() for v in values.values())
        means = {metric: float(v.mean()) for metric, v in values.items()}
        rows.append({'r': r, 'rho': .1, 'label': labels[index], 'means': means,
                     'sample_sd': {metric: float(v.std(ddof=1)) for metric, v in values.items()}})
        source_percent = 100 * means[metrics[0]]
        source_text = f'{source_percent:.3f}' if source_percent >= .001 else r'$<0.001$'
        tex.append(f"{r:g} & {name} & {source_text} & {means[metrics[1]]:.3f} & "
                   f"{100*means[metrics[2]]:.3f} & {means[metrics[3]]:.5f}" + r' \\')
    if r == 4.4:
        tex.append(r'\midrule')
tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
target = args.thesis / 'figures/work2-final/mechanism-table.tex'
target.write_text('\n'.join(tex) + '\n')
provenance = {'source_sha256': hashlib.sha256((root / 'final-report.json').read_bytes()).hexdigest(),
              'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'summaries': rows, 'interpretation': 'Descriptive secondary metrics; no additional significance tests.'}
(root / 'mechanism-descriptive.json').write_text(json.dumps(provenance, indent=2) + '\n')
(root / 'export-mechanism-source.py').write_bytes(Path(__file__).read_bytes())
(args.thesis / 'validation/work2-final-mechanism.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps({'descriptive_method_condition_groups': len(rows)}))
