"""Audit the held-out test and apply its prespecified paired analysis."""
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
from paired_analysis import primary_mean_intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    manifest = json.loads((root / 'manifest.json').read_text())
    done = json.loads((root / 'completion.json').read_text())
    assert manifest['purpose'] == 'final held-out test of frozen selected controllers'
    rows = list(csv.DictReader((root / 'summary.csv').open()))
    assert len(rows) == len(manifest['configs']) == done['completed_runs']
    assert sorted(int(row['run']) for row in rows) == list(range(len(rows)))
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
    groups = defaultdict(dict)
    trajectories = defaultdict(list)
    for row in rows:
        i = int(row['run'])
        record = json.loads((root / f'run-{i:04d}.json').read_text())
        config = record['config']
        assert config == manifest['configs'][i]
        assert row['label'] == manifest['labels'][i]
        assert (config['L'], config['M'], config['state_mode'], config['steps']) == (30, 2, 'reputation', 100000)
        for metric, value in record['tail'].items():
            assert row[metric] == '' if value is None else np.isclose(float(row[metric]), value, rtol=0, atol=1e-12)
        for column, metric in [('whole_cooperation', 'cooperation'), ('whole_mean_abs_ni', 'mean_abs_ni')]:
            assert float(row[column]) == record['whole'][metric]
        with np.load(root / f'run-{i:04d}.npz') as arrays:
            assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == record['q_sha256']
            assert hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest() == record['mask_sha256']
            trajectory = arrays['trajectory'].copy()
        assert trajectory.shape == (1000, 9)
        np.testing.assert_array_equal(trajectory[:, 0], np.arange(100, 100001, 100))
        assert np.isclose(trajectory[:, 1].mean(), record['whole']['cooperation'], rtol=0, atol=1e-10)
        assert np.isclose(trajectory[-200:, 1].mean(), record['tail']['cooperation'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(trajectory[:, 2], 5 * (config['r'] - 1) * trajectory[:, 1], rtol=0, atol=1e-10)
        key = (row['label'], config['r'], config['rho'])
        assert config['seed'] not in groups[key]
        groups[key][config['seed']] = record
        trajectories[key].append(trajectory)
    models = manifest['models']
    labels = [m['label'] for m in models]
    family_names = {'learned': 'Full controller', 'cooperation_gate': 'Fixed C + gate',
                    'selection_only': 'Learned score only', 'gate_only': 'Max R + gate'}
    baseline_names = {'iql': 'IQL', 'ni_global': 'Global NI', 'ni_local': 'Local NI',
                      'trimmed_local': 'Upper-tail filter', 'cooperation_first': 'Fixed C'}
    names = {m['label']: family_names[m['family']] if m.get('family') in family_names else
             f"{baseline_names[m['method']]} (k={m['kappa']:g})" for m in models}
    cells = [(4.4, 0.), (4.4, .1), (4.8, 0.), (4.8, .1)]
    seeds = list(range(1000, 1030))
    full = labels[0]
    assert len(groups) == len(models) * 4
    for label in labels:
        for r, rho in cells:
            assert sorted(groups[(label, r, rho)]) == seeds
            for seed in seeds:
                assert groups[(label, r, rho)][seed]['mask_sha256'] == groups[(full, r, rho)][seed]['mask_sha256']
    contrasts, columns = [], []
    for r, rho in cells:
        for comparison in manifest['primary_comparisons']:
            assert comparison['reference'] == full
            baseline = comparison['baseline']
            values = [groups[(full, r, rho)][seed]['whole']['cooperation'] - groups[(baseline, r, rho)][seed]['whole']['cooperation'] for seed in seeds]
            columns.append(values)
            contrasts.append({'r': r, 'rho': rho, 'baseline': baseline, 'paired_differences': values})
    intervals = primary_mean_intervals(np.array(columns).T)
    for contrast, interval in zip(contrasts, intervals):
        contrast.update(interval)
    report = ['# 冻结模型的最终留出测试', '',
              f'已核验{len(rows)}次运行；每个方法/条件含30个环境种子。',
              '模型参数在测试前冻结，个体Q表重新初始化；全部运行固定100000步，末段为最后20%。',
              '八个主要比较使用同一组种子块进行100000次bootstrap，随机种子20260915。',
              '95%为逐比较百分位区间；99.375%为对八个主要比较作Bonferroni校正的近似区间。',
              '不将点估计排序或个别区间解释为任意环境下的统一优势。', '',
              '| 方法 | r | 异常比例 | 全程均值 | 全程SD | 全程中位数 | 全程范围 | 末段均值 | 末段SD | 末段中位数 | 末段范围 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    descriptive = []
    for label in labels:
        for r, rho in cells:
            records = [groups[(label, r, rho)][seed] for seed in seeds]
            whole = np.array([rec['whole']['cooperation'] for rec in records])
            tail = np.array([rec['tail']['cooperation'] for rec in records])
            descriptive.append({'label': label, 'r': r, 'rho': rho, 'seeds': seeds,
                                'whole': whole.tolist(), 'tail': tail.tolist(),
                                'whole_mean_abs_ni': [rec['whole']['mean_abs_ni'] for rec in records],
                                'whole_metrics': {metric: [rec['whole'][metric] for rec in records] for metric in records[0]['whole']},
                                'tail_metrics': {metric: [rec['tail'][metric] for rec in records] for metric in records[0]['tail']}})
            report.append(f'| {label} | {r:g} | {rho:g} | {whole.mean():.6f} | {whole.std(ddof=1):.6f} | '
                          f'{np.median(whole):.6f} | {whole.min():.6f}–{whole.max():.6f} | {tail.mean():.6f} | '
                          f'{tail.std(ddof=1):.6f} | {np.median(tail):.6f} | {tail.min():.6f}–{tail.max():.6f} |')
    report += ['', '## 八个主要配对比较', '',
               '差值方向为完整模型减去对照，单位为合作率百分点。', '',
               '| r | 异常比例 | 对照 | 均值差 | 95%区间 | 99.375%区间 |',
               '| ---: | ---: | --- | ---: | ---: | ---: |']
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{最终测试的主要全程合作配对差及bootstrap区间，单位为百分点}',
           r'\label{tab:w2-final-primary}', r'\begin{tabular}{cclrrr}', r'\toprule',
           r'$r$ & 异常比例 & 对照 & 均值差 & 95\%区间 & 99.375\%区间 \\', r'\midrule']
    for item in contrasts:
        mean = item['mean_difference'] * 100
        lo, hi = np.array(item['interval_95']) * 100
        family_lo, family_hi = np.array(item['interval_99_375']) * 100
        report.append(f"| {item['r']:g} | {item['rho']:g} | {item['baseline']} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | [{family_lo:.4f}, {family_hi:.4f}] |")
        name = '固定合作规则' if item['baseline'] == 'cooperation_first-0.2' else '固定评分门控'
        tex.append(f"{item['r']:g} & {item['rho']:.0%}".replace('%', r'\%') +
                   f' & {name} & {mean:.3f} & [{lo:.3f}, {hi:.3f}] & [{family_lo:.3f}, {family_hi:.3f}]' + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (root / 'primary-table.tex').write_text('\n'.join(tex) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), layout='constrained', sharey=True)
    for ax, comparison in zip(axes, manifest['primary_comparisons']):
        items = [item for item in contrasts if item['baseline'] == comparison['baseline']]
        for y, item in enumerate(items):
            mean = item['mean_difference'] * 100
            lo, hi = np.array(item['interval_99_375']) * 100
            ax.hlines(y, lo, hi, color='#356fa6', linewidth=3)
            lo, hi = np.array(item['interval_95']) * 100
            ax.hlines(y, lo, hi, color='black', linewidth=1)
            ax.plot(mean, y, 'o', color='#356fa6', markersize=4)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title('Full minus fixed rule' if comparison['baseline'] == 'cooperation_first-0.2' else 'Full minus fixed score + gate', fontsize=10)
        ax.set_yticks(range(4), ['r=4.4, clean', 'r=4.4, 10% fault', 'r=4.8, clean', 'r=4.8, 10% fault'])
        ax.set_xlabel('Whole-cooperation difference (percentage points)')
        ax.grid(axis='x', alpha=.15)
    axes[0].invert_yaxis()
    fig.savefig(root / 'primary-differences.pdf'); fig.savefig(root / 'primary-differences.png', dpi=180)
    plt.close(fig)
    for metric in ['whole', 'tail']:
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 7), layout='constrained', sharex=True, sharey=True)
        for ax, (r, rho) in zip(axes.flat, cells):
            for y, label in enumerate(labels):
                values = np.array([groups[(label, r, rho)][seed][metric]['cooperation'] for seed in seeds])
                # Fixed offsets separate seed markers; their y-coordinate has no metric meaning.
                ax.scatter(values, y + np.linspace(-.15, .15, 30), s=8, alpha=.5)
                ax.plot(values.mean(), y, '|', color='black', markersize=10)
            ax.set_title(f'r={r:g}; faulty fraction={rho:g}', fontsize=10)
            ax.set_yticks(range(len(labels)), [names[label] for label in labels], fontsize=9)
            ax.set_xlim(0, 1.01); ax.set_xlabel(f'{metric.capitalize()} cooperation'); ax.grid(axis='x', alpha=.2)
        axes[0, 0].invert_yaxis()
        fig.savefig(root / f'distributions-{metric}.pdf'); fig.savefig(root / f'distributions-{metric}.png', dpi=180)
        plt.close(fig)
    displayed = [full, labels[1], 'iql-0', 'ni_global-0.5', 'trimmed_local-0.5', 'cooperation_first-0.2']
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout='constrained', sharex=True, sharey=True)
    for ax, (r, rho) in zip(axes.flat, cells):
        for label in displayed:
            curves = np.array(trajectories[(label, r, rho)])
            ax.plot(curves[0, :, 0], curves[:, :, 1].mean(axis=0), label=names[label], linewidth=1)
        ax.set_title(f'r={r:g}; faulty fraction={rho:g}')
        ax.set_xscale('log'); ax.set_ylim(0, 1.01)
        ax.set_xlabel('Step'); ax.set_ylabel('Cooperation')
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncols=3, fontsize=8)
    fig.savefig(root / 'trajectories.pdf'); fig.savefig(root / 'trajectories.png', dpi=180)
    plt.close(fig)
    report += ['', '## 范围与完整性', '',
               '已核验CSV与逐次JSON、Q表及掩码散列、轨迹步号、全程/末段合作与分段轨迹的对应，以及真实福利恒等式。',
               '测试变异对应冻结控制器在不同环境初始化下的运行，不能作为重新训练次数。',
               '全部预定运行均进入统计。并行墙钟时间不用于算法速度排名。',
               '工具来源：Kassis et al. (2026), Scientific Agent Skills, https://doi.org/10.48550/arXiv.2609.00065 。']
    (root / 'REPORT.md').write_text('\n'.join(report) + '\n')
    result = {'verified_runs': len(rows), 'descriptive': descriptive, 'primary_comparisons': contrasts,
              'bootstrap_resamples': 100000, 'bootstrap_seed': 20260915,
              'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ['manifest.json', 'summary.csv']},
              'analysis_sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name('paired_analysis.py')]}}
    (root / 'final-report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for p in [Path(__file__), Path(__file__).with_name('paired_analysis.py')]:
        (root / ('analysis-' + p.name)).write_bytes(p.read_bytes())
    print(json.dumps({'verified_runs': len(rows), 'primary_comparisons': len(contrasts)}))


if __name__ == '__main__':
    main()
