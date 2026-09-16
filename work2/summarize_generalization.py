"""Audit frozen-model development suites and report independent-run variation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CELL_KEYS = ('L', 'M', 'state_mode', 'r', 'rho', 'attack')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / 'manifest.json').read_text())
    completion = json.loads((root / 'completion.json').read_text())
    rows = list(csv.DictReader((root / 'summary.csv').open()))
    assert len(rows) == len(manifest['configs']) == completion['completed_runs']
    assert sorted(int(row['run']) for row in rows) == list(range(len(rows)))
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
    groups = defaultdict(dict)
    for row in rows:
        index = int(row['run'])
        record = json.loads((root / f'run-{index:04d}.json').read_text())
        config = record['config']
        assert config == manifest['configs'][index]
        assert row['label'] == manifest['labels'][index]
        for key in CELL_KEYS + ('seed', 'steps', 'method', 'kappa'):
            assert row[key] == str(config[key])
        for metric, value in record['tail'].items():
            assert row[metric] == '' if value is None else np.isclose(float(row[metric]), value, rtol=0, atol=1e-12)
        assert np.isclose(float(row['whole_cooperation']), record['whole']['cooperation'], rtol=0, atol=1e-12)
        with np.load(root / f'run-{index:04d}.npz') as arrays:
            assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == record['q_sha256']
            assert hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest() == record['mask_sha256']
            assert arrays['actions'].shape == (config['L'] ** 2,)
            assert np.isfinite(arrays['q']).all()
        cell = tuple(config[key] for key in CELL_KEYS)
        bucket = groups[(cell, row['label'])]
        assert config['seed'] not in bucket
        bucket[config['seed']] = record
    models = [model['label'] for model in manifest['models']]
    cells = list(dict.fromkeys(cell for cell, _ in groups))
    names = ['Learned score + gate', 'Fixed C score + gate', 'IQL',
             'Global NI', 'Upper-tail filter', 'Fixed C score + strength']
    # The runner freezes this ordering before any new environment is evaluated.
    assert len(models) == len(names)
    report = ['# 冻结控制器的泛化开发报告', '',
              f"实验组：{manifest['suite']}；已核验 {len(rows)} 次完整运行。",
              f"执行步数：{sorted({config['steps'] for config in manifest['configs']})}。",
              '本批属于开发验证，不是最终留出测试；控制器由先前结构验证固定，未按本批条件重选或重训。',
              '全程均值使用全部执行步，末段使用最后20%；固定步数，无提前停止。',
              '重复单位为独立仿真。逐条件报告均值、样本标准差与范围，不以节点或时间步扩充样本量。', '',
              '| 模型 | L | M | 状态 | r | 异常比例 | 消息 | n | 全程均值 | 末段均值 | 末段SD | 末段范围 |',
              '| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    if any(config['steps'] != 100000 for config in manifest['configs']):
        report.insert(3, '**本批为短运行流程检查，以下数值不构成研究结果。**')
    summaries, differences = [], []
    fig, axes = plt.subplots(int(np.ceil(len(cells) / 4)), 4,
                             figsize=(15, 3.1 * int(np.ceil(len(cells) / 4))),
                             layout='constrained', squeeze=False)
    for ax, cell in zip(axes.flat, cells):
        reference = groups[(cell, models[0])]
        for j, label in enumerate(models):
            runs = groups[(cell, label)]
            assert runs.keys() == reference.keys()
            seeds = sorted(runs)
            for seed in seeds:
                assert runs[seed]['mask_sha256'] == reference[seed]['mask_sha256']
            whole = np.array([runs[s]['whole']['cooperation'] for s in seeds])
            tail = np.array([runs[s]['tail']['cooperation'] for s in seeds])
            sd = float(tail.std(ddof=1)) if len(tail) > 1 else None
            summaries.append({'condition': dict(zip(CELL_KEYS, cell)), 'label': label,
                              'seeds': seeds, 'whole': whole.tolist(), 'tail': tail.tolist()})
            sd_text = f'{sd:.5f}' if sd is not None else 'NA'
            report.append(f'| {label} | ' + ' | '.join(map(str, cell)) +
                          f' | {len(seeds)} | {whole.mean():.5f} | {tail.mean():.5f} | {sd_text} | {tail.min():.5f}–{tail.max():.5f} |')
            ax.scatter(np.full(len(tail), j), tail, s=12, alpha=.65)
            ax.plot(j, tail.mean(), marker='_', color='black', markersize=13)
            if label != models[0]:
                differences.append({'condition': dict(zip(CELL_KEYS, cell)), 'baseline': label,
                                    'seeds': seeds,
                                    'whole_differences': [reference[s]['whole']['cooperation'] - runs[s]['whole']['cooperation'] for s in seeds],
                                    'tail_differences': [reference[s]['tail']['cooperation'] - runs[s]['tail']['cooperation'] for s in seeds]})
        side, radius, state, r, rho, attack = cell
        ax.set_title(f'L={side}, M={radius}, {state}\nr={r:g}, rho={rho:g}, {attack}', fontsize=9)
        ax.set_xticks(range(len(models)), ['Full', 'C+gate', 'IQL', 'Global', 'Trim', 'C+fixed'], rotation=45, fontsize=8)
        ax.set_ylim(0, 1.01)
        ax.set_ylabel('Tail cooperation')
    for ax in list(axes.flat)[len(cells):]:
        ax.set_visible(False)
    fig.suptitle('Frozen controllers: development generalization; points = independent runs')
    fig.savefig(root / 'generalization.pdf')
    fig.savefig(root / 'generalization.png', dpi=140)
    plt.close(fig)
    report += ['', '## 完整模型减去对照的配对差', '',
               '| 条件 (L,M,状态,r,比例,消息) | 对照 | 全程差均值 | 末段差均值 | 末段差范围 |',
               '| --- | --- | ---: | ---: | ---: |']
    for item in differences:
        values = item['tail_differences']
        report.append(f"| {tuple(item['condition'].values())} | {item['baseline']} | "
                      f"{np.mean(item['whole_differences']):.5f} | {np.mean(values):.5f} | {min(values):.5f}–{max(values):.5f} |")
    report += ['', '异常来源选择率按固定发送者身份统计；间歇性发送者在如实报告期间仍属于该身份。',
               '合法区间内的虚假消息不一定降低合作；本实验不把所有消息失真预设为同等有害。',
               '不同网格规模的个体不一一对应，配对只在相同条件的算法之间进行。',
               '当前并发墙钟时间不构成算法开销比较。最终测试种子1000–1029未用于本批。', '',
               '工具来源：Kassis et al. (2026), Scientific Agent Skills, https://doi.org/10.48550/arXiv.2609.00065 。']
    (root / 'REPORT.md').write_text('\n'.join(report) + '\n')
    result = {'verified_runs': len(rows), 'suite': manifest['suite'], 'summaries': summaries,
              'comparisons': differences, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'generalization.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (root / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'verified_runs': len(rows), 'conditions': len(cells), 'suite': manifest['suite']}))


if __name__ == '__main__':
    main()
