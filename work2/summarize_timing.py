"""Audit all prespecified timing repetitions and export their cost/quality table."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--thesis', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / 'manifest.json').read_text())
    done = json.loads((root / 'completion.json').read_text())
    assert done['completed_runs'] == len(manifest['configs']) == 60
    assert len(manifest['timing_cpu_affinity']) == 1
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
    groups = defaultdict(dict)
    for i, config in enumerate(manifest['configs']):
        rec = json.loads((root / f'run-{i:04d}.json').read_text())
        assert config == rec['config']
        assert rec['label'] == manifest['labels'][i]
        assert (config['M'], config['r'], config['rho'], config['steps'], config['attack']) == (2, 4.8, .1, 10000, 'high_defect')
        assert config['seed'] in [220, 221, 222] and config['L'] in [30, 100]
        assert rec['wall_seconds'] > 0 and rec['cpu_seconds'] > 0
        with np.load(root / f'run-{i:04d}.npz') as arrays:
            assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == rec['q_sha256']
            assert hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest() == rec['mask_sha256']
            assert np.isfinite(arrays['q']).all()
            assert arrays['trajectory'].shape == (100, 9)
            assert np.isclose(arrays['trajectory'][:, 1].mean(), rec['whole']['cooperation'], rtol=0, atol=1e-10)
        key = (rec['label'], config['L'])
        assert config['seed'] not in groups[key]
        groups[key][config['seed']] = rec
    labels = [m['label'] for m in manifest['models']]
    assert len(groups) == 20
    names = ['完整模型', '固定评分/学习门控', '学习评分/固定门控', '最高奖励/学习门控',
             '独立Q-learning', '全局NI', '局部NI', '上尾过滤', '固定合作规则', '幅度校准固定规则']
    rows = []
    report = ['# 单进程执行开销', '',
              '全部60次计时均保留。固定单核、预热、随机化方法顺序；相同规模/种子下方法配对。',
              '相同10000步预算，包含初始化和统计、不含文件序列化和首次JIT预热。',
              '机器为共享环境；报告每次CPU时间和墙钟，不将本表写成同等合作质量下的加速比。', '',
              '| L | 方法 | 每步墙钟均值(ms) | 中位数 | 范围 | 每步CPU均值(ms) | 全程合作均值 |',
              '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{同预算执行开销与合作水平；$r=4.8$、10\%受扰、每项三次10000步运行}',
           r'\label{tab:w2-timing}', r'\begin{tabular}{clrrrr}', r'\toprule',
           r'$L$ & 方法 & 每步墙钟/ms & 墙钟范围/ms & 每步CPU/ms & 全程合作 \\', r'\midrule']
    for side in [30, 100]:
        for label, name in zip(labels, names):
            runs = groups[label, side]
            assert sorted(runs) == [220, 221, 222]
            for seed in runs:
                assert runs[seed]['mask_sha256'] == groups[labels[0], side][seed]['mask_sha256']
            wall = np.array([runs[s]['wall_seconds']/10 for s in sorted(runs)])
            cpu = np.array([runs[s]['cpu_seconds']/10 for s in sorted(runs)])
            cooperation = np.array([runs[s]['whole']['cooperation'] for s in sorted(runs)])
            rows.append({'L': side, 'label': label, 'seeds': sorted(runs),
                         'wall_ms_per_step': wall.tolist(), 'cpu_ms_per_step': cpu.tolist(),
                         'whole_cooperation': cooperation.tolist()})
            report.append(f'| {side} | {label} | {wall.mean():.4f} | {np.median(wall):.4f} | '
                          f'{wall.min():.4f}–{wall.max():.4f} | {cpu.mean():.4f} | {cooperation.mean():.5f} |')
            tex.append(f'{side} & {name} & {wall.mean():.3f} & {wall.min():.3f}--{wall.max():.3f} & '
                       f'{cpu.mean():.3f} & {cooperation.mean():.4f}' + r' \\')
        if side == 30: tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip\footnotesize 数据来源：本文单进程计时实验，2026年。', r'\end{table}']
    report += ['', '每次值、运行顺序与前后负载均保留。仅三个重复，不进行耗时显著性推断。',
               '可训练参数少不等于整个仿真耗时接近独立Q-learning；候选排序、特征和门控均产生执行成本。',
               '工具来源：沿用 RESEARCH_PLAN.md 中实验设计和统计分析技能来源。']
    (root / 'REPORT.md').write_text('\n'.join(report) + '\n')
    audit = {'verified_runs': 60, 'summaries': rows,
             'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ['manifest.json', 'completion.json', 'hardware.txt']},
             'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'timing.json').write_text(json.dumps(audit, indent=2) + '\n')
    (root / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    dst = args.thesis / 'figures/work2-timing'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'timing-table.tex').write_text('\n'.join(tex) + '\n')
    (args.thesis / 'validation/work2-timing-sources.json').write_text(json.dumps({str(root.resolve()): hashlib.sha256((root / 'timing.json').read_bytes()).hexdigest()}, indent=2) + '\n')
    print(json.dumps({'verified_runs': 60, 'method_size_groups': len(rows)}))


if __name__ == '__main__': main()
