"""Matched validation of separately retrained selection and gate ablations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    manifest = json.loads((root / 'manifest.json').read_text())
    selected = json.loads((root / 'selection.json').read_text())
    rows = list(csv.DictReader((root / 'summary.csv').open()))
    complete = json.loads((root / 'completion.json').read_text())
    assert len(rows) == len(manifest['configs']) == complete['completed_runs']
    families = selected['selected_by_family']
    assert {'learned', 'cooperation_gate'} <= families.keys()
    for family, label in families.items():
        candidates = [m['label'] for m in manifest['models'] if m.get('family') == family]
        assert label == max(candidates, key=selected['scores'].get)
    full = families['learned']
    family_names = {'learned': '完整模型', 'cooperation_gate': '固定合作评分',
                    'selection_only': '仅学习评分', 'gate_only': '最高奖励评分'}
    plot_names = {'learned': 'Learned score\nand gate',
                  'cooperation_gate': 'Cooperative score\nlearned gate',
                  'selection_only': 'Learned score\nfixed gate',
                  'gate_only': 'Max-reward score\nlearned gate'}
    order = [f for f in ['gate_only', 'selection_only', 'cooperation_gate', 'learned'] if f in families]
    assert set(order) == set(families)
    table = {(r['label'], float(r['r']), float(r['rho']), int(r['seed'])): r for r in rows}
    cells = sorted({(float(r['r']), float(r['rho'])) for r in rows})
    seeds = sorted({int(r['seed']) for r in rows})
    assert len(table) == len(rows)
    report = ['# 邻居选择与门控结构的共同验证', '',
              '本批只用于开发与模型选择，不能作为最终留出测试。',
              '每类方法有三个优化初始化，每次输出两个候选；六个候选不是六次独立训练。',
              '各类方法使用相同候选评价预算和训练种子池，逐代训练环境没有配对；验证按环境种子配对。',
              '仅学习评分的模型固定门控为0.4；其余学习模型训练门控，均使用最大NI强度0.5。', '',
              '| 模型族 | 所选候选 | 全部候选全程目标范围 | 所选全程目标 |',
              '| --- | --- | ---: | ---: |']
    for family in order:
        label = families[family]
        values = [selected['scores'][m['label']] for m in manifest['models'] if m.get('family') == family]
        report.append(f'| {family_names[family]} | {label} | {min(values):.6f}–{max(values):.6f} | {selected["scores"][label]:.6f} |')
    report += ['', '差值方向为完整模型减去对应消融模型，末段取最后20%。', '',
               '| 消融模型 | r | 异常比例 | n | 全程配对差均值 | 末段配对差均值 | 末段配对差范围 |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, layout='constrained')
    comparisons = []
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{完整模型相对重新训练消融模型的验证配对差；正值表示完整模型较高}',
           r'\label{tab:w2-score-ablation}', r'\begin{tabular}{lccrr}', r'\toprule',
           r'消融模型 & $r$ & 异常比例 & 全程差均值 & 末段差均值 \\', r'\midrule']
    for ax, (r, rho) in zip(axes.flat, cells):
        for family in order:
            if family == 'learned':
                continue
            other = families[family]
            whole = [float(table[(full, r, rho, s)]['whole_cooperation']) -
                     float(table[(other, r, rho, s)]['whole_cooperation']) for s in seeds]
            tail = [float(table[(full, r, rho, s)]['cooperation']) -
                    float(table[(other, r, rho, s)]['cooperation']) for s in seeds]
            report.append(f'| {family_names[family]} | {r:g} | {rho:g} | {len(seeds)} | {np.mean(whole):.6f} | '
                          f'{np.mean(tail):.6f} | {min(tail):.6f}–{max(tail):.6f} |')
            comparisons.append({'family': family, 'candidate': other, 'r': r, 'rho': rho,
                                'seeds': seeds, 'whole_paired_differences': whole,
                                'tail_paired_differences': tail})
            tex.append(f'{family_names[family]} & {r:g} & {rho:.0%}'.replace('%',r'\%') +
                       f' & {np.mean(whole):.5f} & {np.mean(tail):.5f}' + r' \\')
        labels = ['cooperation_first-0.2'] + [families[f] for f in order]
        for seed in seeds:
            values = [float(table[(label, r, rho, seed)]['whole_cooperation']) for label in labels]
            ax.plot(range(len(labels)), values, 'o-', alpha=.65, label=str(seed), linewidth=1)
        ax.set_xticks(range(len(labels)), ['Fixed rule\n(k=0.2)'] + [plot_names[f] for f in order], fontsize=8)
        ax.set_title(f'r={r:g}, faulty senders={rho:.0%}')
        ax.set_ylabel('Whole-trajectory cooperation'); ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(title='Environment seed', fontsize=8)
    fig.suptitle('Model-selection data; each line is one matched environment seed')
    fig.savefig(root / 'score-ablation.pdf'); fig.savefig(root / 'score-ablation.png', dpi=160)
    plt.close(fig)
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (root / 'score-ablation-table.tex').write_text('\n'.join(tex) + '\n')
    report += ['', '图中各点为完整仿真的全程平均；连线仅标识配对种子，不表示连续算法参数。',
               '没有把时间步或网格节点当作独立样本，也不据当前选型数据宣称统计显著。',
               '更多攻击、状态观测及尺度下的有效性需另行检验；当前对照不证明一般消息真假识别。']
    (root / 'SCORE_ABLATION.md').write_text('\n'.join(report) + '\n')
    result = {'selected_by_family': families, 'comparisons': comparisons,
              'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ['manifest.json', 'selection.json', 'summary.csv']},
              'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'score-ablation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('\n'.join(report[:19]))


if __name__ == '__main__':
    main()
