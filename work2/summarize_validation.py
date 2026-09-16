"""Report every validation candidate; model selection is not a final test."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / 'manifest.json').read_text())
    completion = json.loads((root / 'completion.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    rows = list(csv.DictReader((root / 'summary.csv').open()))
    assert len(rows) == len(manifest['configs']) == completion['completed_runs']
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
    groups = defaultdict(list)
    for row in rows:
        index = int(row['run'])
        record = json.loads((root / f'run-{index:04d}.json').read_text())
        assert record['config'] == manifest['configs'][index]
        assert row['label'] == manifest['labels'][index]
        for metric, value in record['tail'].items():
            if value is None:
                assert row[metric] == ''
            else:
                assert np.isclose(float(row[metric]), value, rtol=0, atol=1e-12)
        assert np.isclose(float(row['whole_cooperation']), record['whole']['cooperation'], rtol=0, atol=1e-12)
        with np.load(root / f'run-{index:04d}.npz') as arrays:
            assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == record['q_sha256']
            assert hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest() == record['mask_sha256']
        groups[(row['label'], float(row['r']), float(row['rho']))].append(row)
    report = ['# 仿真批次描述性报告', '',
              f"批次目的：{manifest['purpose']}。",
              '本批属于开发或模型验证，不是最终留出测试。',
              '合作率末段为最后20%时间窗口；全程均值用于模型选择。没有提前停止。',
              '逐次运行是重复单位，以下同时报告均值、样本标准差与范围，不作显著性判断。', '',
              f"本批用于配对绘图的候选：`{selection['selected_candidate']}`。", '',
              '| 模型 | 四单元全程等权均值 |', '| --- | ---: |']
    for model in manifest['models']:
        label = model['label']
        calculated = np.mean([float(r['whole_cooperation']) for r in rows if r['label'] == label])
        assert np.isclose(calculated, selection['scores'][label], rtol=0, atol=1e-12)
        report.append(f'| {label} | {calculated:.6f} |')
    report += ['', '## 全部候选与基线', '',
               '| 模型 | r | 异常比例 | n | 全程均值 | 末段均值 | 末段SD | 末段范围 |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (label, r, rho), cell in groups.items():
        values = np.array([float(row['cooperation']) for row in cell])
        whole = np.mean([float(row['whole_cooperation']) for row in cell])
        report.append(f'| {label} | {r:g} | {rho:g} | {len(cell)} | {whole:.4f} | '
                      f'{values.mean():.4f} | {values.std(ddof=1):.4f} | {values.min():.4f}–{values.max():.4f} |')
    chosen = selection['selected_candidate']
    baselines = [m['label'] for m in manifest['models'] if m['controller'] is None]
    method_names = {'iql':'IQL','ni_global':'Global NI','ni_local':'Local NI',
                    'trimmed_local':'Upper-tail filter','cooperation_first':'Cooperation first',
                    'cooperation_only':'Cooperation direction only'}
    plot_names = {m['label']:f"{method_names[m['method']]} (k={m['kappa']:g})"
                  for m in manifest['models'] if m['controller'] is None}
    plot_names[chosen] = 'Untrained semantic prior' if chosen == 'semantic-prior' else 'Learned controller'
    report += ['', '## 所选候选相对基线的配对差', '',
               '差值为所选候选减去基线，按环境种子配对；这些差值仍属于模型选择数据。', '',
               '| 基线 | r | 异常比例 | 全程差均值 | 末段差均值 | 末段差范围 |',
               '| --- | ---: | ---: | ---: | ---: | ---: |']
    cells = sorted({(float(row['r']), float(row['rho'])) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout='constrained')
    for ax, (r, rho) in zip(axes.flat, cells):
        selected_rows = {int(row['seed']): row for row in groups[(chosen, r, rho)]}
        for baseline in baselines:
            base_rows = {int(row['seed']): row for row in groups[(baseline, r, rho)]}
            assert base_rows.keys() == selected_rows.keys()
            tail = [float(selected_rows[s]['cooperation']) - float(base_rows[s]['cooperation']) for s in base_rows]
            whole = [float(selected_rows[s]['whole_cooperation']) - float(base_rows[s]['whole_cooperation']) for s in base_rows]
            report.append(f'| {baseline} | {r:g} | {rho:g} | {np.mean(whole):.4f} | '
                          f'{np.mean(tail):.4f} | {min(tail):.4f}–{max(tail):.4f} |')
        for label in baselines + [chosen]:
            curves = []
            for row in groups[(label, r, rho)]:
                with np.load(root / f"run-{int(row['run']):04d}.npz") as data:
                    trajectory = data['trajectory']
                    curves.append(trajectory[:, 1])
            ax.plot(trajectory[:, 0], np.mean(curves, axis=0), label=plot_names[label],
                    linewidth=2 if label == chosen else 1)
        ax.set_title(f'r={r:g}; faulty fraction={rho:g}')
        ax.set_xscale('log'); ax.set_ylim(0, 1); ax.set_xlabel('Step'); ax.set_ylabel('Cooperation')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, fontsize=8)
    fig.suptitle('Development / validation trajectories: mean of independent runs')
    fig.savefig(root / 'trajectories.pdf'); fig.savefig(root / 'trajectories.png', dpi=160)
    plt.close(fig)
    # Show the first validation seed by a fixed rule, not the most favorable snapshot.
    snapshot_seed = min(int(row['seed']) for row in rows)
    snapshot_labels = ['ni_global-0.5', 'trimmed_local-0.5', 'cooperation_first-0.5', chosen]
    rs = sorted({float(row['r']) for row in rows})
    fig, axes = plt.subplots(len(rs), len(snapshot_labels), figsize=(10, 5.7), layout='constrained', squeeze=False)
    for row_index, r in enumerate(rs):
        for column, label in enumerate(snapshot_labels):
            record = next(row for row in groups[(label, r, .1)] if int(row['seed']) == snapshot_seed)
            with np.load(root / f"run-{int(record['run']):04d}.npz") as arrays:
                actions = arrays['actions']
                side = int(np.sqrt(len(actions)))
                ax = axes[row_index, column]
                ax.imshow(actions.reshape(side, side), cmap=ListedColormap(['#2478b4', '#dddddd']), vmin=0, vmax=1)
                ax.set_title(f'{plot_names[label]}\nr={r:g}; final C={np.mean(actions == 0):.3f}', fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Development / validation snapshot, seed {snapshot_seed}, 10% faulty senders; blue C / gray D')
    fig.savefig(root / 'snapshots.pdf'); fig.savefig(root / 'snapshots.png', dpi=160)
    plt.close(fig)
    report += ['', '## 完整性与边界', '',
               f"已核验{len(rows)}个运行配置、Q表及异常掩码散列，源码快照散列匹配。",
               '全部候选与基线均保留；所绘制候选不代表所有模型均达到该表现。',
               '平均曲线不显示所有运行间差异，应与上表逐条件范围一起阅读。',
               f'空间快照固定采用本批最小种子{snapshot_seed}，展示末轮真实动作，既非末段平均也非异常身份。',
               '并行工作进程耗时受本机并发负载影响，不作为公平算法开销结论。',
               '最终测试种子仍保留；本报告不能证明未见参数、攻击或网络下的泛化。', '',
               '工具来源：Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, '
               'https://doi.org/10.48550/arXiv.2609.00065 。']
    (root / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'verified_runs': len(rows), 'selected': chosen, 'scores': selection['scores']}, indent=2))


if __name__ == '__main__':
    main()
