"""Export audited message/parameter development results to the thesis."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--messages', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--scale', type=Path, required=True)
    parser.add_argument('--thesis', type=Path, required=True)
    args = parser.parse_args()
    sources = [args.messages, args.parameters, args.state, args.scale]
    records, provenance = [], {}
    for root in sources:
        audit = json.loads((root / 'generalization.json').read_text())
        manifest = json.loads((root / 'manifest.json').read_text())
        assert audit['verified_runs'] == len(manifest['configs'])
        assert all(c['steps'] == 100000 for c in manifest['configs'])
        records.extend(audit['summaries'])
        provenance[str(root.resolve())] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                          for name in ['manifest.json', 'summary.csv', 'generalization.json']}
    models = [m['label'] for m in json.loads((args.messages / 'manifest.json').read_text())['models']]
    assert models == [m['label'] for m in json.loads((args.parameters / 'manifest.json').read_text())['models']]
    lookup = {(x['label'], x['condition']['r'], x['condition']['rho'], x['condition']['attack'],
               x['condition']['M'], x['condition']['state_mode'], x['condition']['L']): x for x in records}
    assert len(lookup) == len(records)
    def values(label, r, rho, attack, metric='tail', M=2, state='reputation', L=30):
        record = lookup[(label, r, rho, attack, M, state, L)]
        assert record['seeds'] == [216, 217, 218, 219]
        return np.array(record[metric])
    dst = args.thesis / 'figures/work2-generalization'
    dst.mkdir(parents=True, exist_ok=True)
    attacks = ['none', 'high_defect', 'high_only', 'flip_action', 'random_message',
               'burst_defect', 'moderate_defect', 'high_cooperate']
    chinese = ['干净', '高奖励背叛', '仅抬高奖励', '仅翻转动作', '随机消息',
               '间歇背叛', '较小抬高', '高奖励合作']
    names = ['Full controller', 'Fixed C score + gate', 'IQL', 'Global NI',
             'Upper-tail filter', 'Fixed C score + strength']
    display_order = [2, 3, 4, 5, 1, 0]
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{不同消息条件下的末段合作均值；每格为四次开发验证的平均}',
           r'\label{tab:w2-message-generalization}', r'\begin{tabular}{clrrrrr}',
           r'\toprule', r'$r$ & 条件 & 全局NI & 上尾过滤 & 固定规则 & 固定评分/门控 & 完整模型 \\', r'\midrule']
    for r in [4.4, 4.8]:
        for attack, name in zip(attacks, chinese):
            rho = 0. if attack == 'none' else .1
            means = [values(models[i], r, rho, attack).mean() for i in [3, 4, 5, 1, 0]]
            tex.append(f'{r:g} & {name} & ' + ' & '.join(f'{x:.4f}' for x in means) + r' \\')
        if r == 4.4: tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (dst / 'messages-table.tex').write_text('\n'.join(tex) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 6.4), layout='constrained', sharey=True)
    ylabels = ['Clean', 'High D', 'High R only', 'Flip action', 'Random', 'Burst D', 'Moderate D', 'High C']
    colors = plt.get_cmap('tab10').colors
    for ax, r in zip(axes, [4.4, 4.8]):
        for j, i in enumerate(display_order):
            y = np.arange(len(attacks)) + (j - 2.5) * .105
            means = []
            for a, yy in zip(attacks, y):
                vals = values(models[i], r, 0. if a == 'none' else .1, a)
                ax.scatter(vals, np.full(4, yy), color=colors[i], s=9, alpha=.35)
                means.append(vals.mean())
            ax.scatter(means, y, color=colors[i], s=26, marker='|', label=names[i])
        ax.set_title(f'r={r:g}', fontsize=12)
        ax.set_xlabel('Tail cooperation', fontsize=11)
        ax.set_xlim(-.015, 1.015)
        ax.set_yticks(range(8), ylabels, fontsize=10)
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncols=2, fontsize=9)
    fig.savefig(dst / 'messages.pdf'); fig.savefig(dst / 'messages.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True, layout='constrained')
    for ax, rho in zip(axes[:2], [0., .1]):
        rs = [3., 3.6, 4., 4.2, 4.4, 4.6, 4.8]
        for i in display_order:
            means = [values(models[i], r, rho, 'high_defect' if rho else 'none').mean() for r in rs]
            ax.plot(rs, means, '.-', color=colors[i], label=names[i], markersize=5, linewidth=1)
        ax.set_title(f'Faulty fraction={rho:g}')
        ax.set_xlabel('Synergy factor r')
    for i in display_order:
        rhos = [0., .05, .1, .2, .3, .5]
        means = [values(models[i], 4.8, rho, 'high_defect' if rho else 'none').mean() for rho in rhos]
        axes[2].plot(rhos, means, '.-', color=colors[i], label=names[i], markersize=5, linewidth=1)
    axes[2].set_title('r=4.8'); axes[2].set_xlabel('Faulty fraction')
    axes[0].set_ylabel('Tail cooperation')
    for ax in axes:
        ax.set_ylim(0, 1.01); ax.grid(alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncols=3, fontsize=8)
    fig.savefig(dst / 'parameters.pdf'); fig.savefig(dst / 'parameters.png', dpi=180)
    plt.close(fig)
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{部分泛化边界的逐次末段合作范围；四次运行全部保留}',
           r'\label{tab:w2-parameter-boundaries}', r'\begin{tabular}{cclrr}', r'\toprule',
           r'$r$ & 异常比例 & 方法 & 末段均值 & 末段范围 \\', r'\midrule']
    for r, rho in [(3., 0.), (3.6, .1), (4.8, .5)]:
        for i, name in [(0, '完整模型'), (1, '固定评分/学习门控'), (5, '固定合作规则')]:
            vals = values(models[i], r, rho, 'high_defect' if rho else 'none')
            tex.append(f'{r:g} & {rho:.0%}'.replace('%', r'\%') + f' & {name} & {vals.mean():.5f} & {vals.min():.5f}--{vals.max():.5f}' + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (dst / 'parameter-boundaries-table.tex').write_text('\n'.join(tex) + '\n')
    state_conditions = [('reputation', 1), ('reputation', 2), ('own_action', 1), ('own_action', 2)]
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{状态与感知条件变化下的末段合作均值；控制器参数保持冻结}',
           r'\label{tab:w2-state-generalization}', r'\begin{tabular}{lcccrrr}', r'\toprule',
           r'状态 & $M$ & $r$ & 异常比例 & 完整模型 & 固定评分/门控 & 固定规则 \\', r'\midrule']
    for r in [4.4, 4.8]:
        for rho in [0., .1]:
            for state, radius in state_conditions:
                name = '声誉' if state == 'reputation' else '自身动作'
                means = [values(models[i], r, rho, 'high_defect' if rho else 'none', M=radius, state=state).mean() for i in [0, 1, 5]]
                tex.append(f'{name} & {radius} & {r:g} & {rho:.0%}'.replace('%', r'\%') +
                           ' & ' + ' & '.join(f'{x:.4f}' for x in means) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (dst / 'state-table.tex').write_text('\n'.join(tex) + '\n')
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.3), layout='constrained', sharex=True, sharey=True)
    for ax, (r, rho) in zip(axes.flat, [(4.4, 0.), (4.4, .1), (4.8, 0.), (4.8, .1)]):
        for j, i in enumerate([2, 5, 1, 0]):
            means = []
            y = np.arange(4) + (j - 1.5) * .13
            for (state, radius), yy in zip(state_conditions, y):
                vals = values(models[i], r, rho, 'high_defect' if rho else 'none', M=radius, state=state)
                ax.scatter(vals, np.full(4, yy), color=colors[i], s=10, alpha=.4)
                means.append(vals.mean())
            ax.scatter(means, y, color=colors[i], s=28, marker='|', label=names[i])
        ax.set_title(f'r={r:g}; faulty fraction={rho:g}', fontsize=11)
        ax.set_yticks(range(4), ['Reputation, M=1', 'Reputation, M=2', 'Own action, M=1', 'Own action, M=2'], fontsize=9)
        ax.set_xlim(0., 1.01); ax.set_xlabel('Tail cooperation', fontsize=10); ax.grid(axis='x', alpha=.2)
    axes[0, 0].invert_yaxis()
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncols=2, fontsize=9)
    fig.savefig(dst / 'state.pdf'); fig.savefig(dst / 'state.png', dpi=180)
    plt.close(fig)
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{不同网格规模的末段合作均值；$r=4.8$，每个条件四次运行}',
           r'\label{tab:w2-scale}', r'\begin{tabular}{ccrrrrr}', r'\toprule',
           r'$L$ & 异常比例 & IQL & 全局NI & 固定规则 & 固定评分/门控 & 完整模型 \\', r'\midrule']
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), layout='constrained', sharey=True)
    for ax, rho in zip(axes, [0., .1]):
        for side in [20, 30, 50, 100]:
            means = [values(models[i], 4.8, rho, 'high_defect' if rho else 'none', L=side).mean() for i in [2, 3, 5, 1, 0]]
            tex.append(f'{side} & {rho:g} & ' + ' & '.join(f'{x:.4f}' for x in means) + r' \\')
        for i in display_order:
            means = []
            for side in [20, 30, 50, 100]:
                vals = values(models[i], 4.8, rho, 'high_defect' if rho else 'none', L=side)
                ax.scatter(np.full(4, side), vals, s=9, alpha=.35, color=colors[i])
                means.append(vals.mean())
            ax.plot([20, 30, 50, 100], means, '.-', color=colors[i], label=names[i], linewidth=1)
        ax.set_title(f'Faulty fraction={rho:g}'); ax.set_xlabel('Grid side L')
        ax.set_ylim(0, 1.01); ax.set_xticks([20, 30, 50, 100]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Tail cooperation')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncols=2, fontsize=8)
    fig.savefig(dst / 'scale.pdf'); fig.savefig(dst / 'scale.png', dpi=180); plt.close(fig)
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (dst / 'scale-table.tex').write_text('\n'.join(tex) + '\n')
    artifact = {'purpose': 'frozen-controller development generalization, not final test',
                'sources': provenance, 'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'exported_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir()}}
    (args.thesis / 'validation/work2-generalization-sources.json').write_text(json.dumps(artifact, indent=2) + '\n')
    (args.messages / 'export-analysis-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'exported_files': list(artifact['exported_files']), 'audited_runs': len(records) * 4}))


if __name__ == '__main__':
    main()
