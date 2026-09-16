"""Audit completed amplitude refinement and frozen-parameter feature removal."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path):
    rec = json.loads(path.read_text())
    with np.load(path.with_suffix('.npz')) as arrays:
        assert hashlib.sha256(arrays['q'].tobytes()).hexdigest() == rec['q_sha256']
        assert hashlib.sha256(arrays['bad_mask'].tobytes()).hexdigest() == rec['mask_sha256']
        assert np.isfinite(arrays['q']).all()
        trajectory = arrays['trajectory']
        assert trajectory.shape == (1000, 9)
        assert np.array_equal(trajectory[:, 0], np.arange(100, 100001, 100))
        assert np.isclose(trajectory[:, 1].mean(), rec['whole']['cooperation'], rtol=0, atol=1e-10)
        assert np.isclose(trajectory[-200:, 1].mean(), rec['tail']['cooperation'], rtol=0, atol=1e-10)
    return rec


def new_runs(root, count):
    manifest = json.loads((root / 'manifest.json').read_text())
    done = json.loads((root / 'completion.json').read_text())
    assert done['completed_runs'] == len(manifest['configs']) == count
    for name, sha in manifest['sources'].items():
        assert digest(root / 'source' / name) == sha
    records = []
    for i, config in enumerate(manifest['configs']):
        rec = read_run(root / f'run-{i:04d}.json')
        assert rec['config'] == config
        assert config['L'] == 30 and config['M'] == 2 and config['steps'] == 100000
        records.append(rec)
    return manifest, records


def summaries(records, names, reference, seeds):
    lookup = {(name, rec['config']['r'], rec['config']['rho'], rec['config']['seed']): rec
              for name, rec in zip(names, records)}
    assert len(lookup) == len(records)
    rows = []
    for r in [4.4, 4.8]:
        for rho in [0., .1]:
            for name in dict.fromkeys(names):
                runs = [lookup[name, r, rho, seed] for seed in seeds]
                refs = [lookup[reference, r, rho, seed] for seed in seeds]
                assert all(x['mask_sha256'] == y['mask_sha256'] for x, y in zip(runs, refs))
                rows.append({'name': name, 'r': r, 'rho': rho, 'seeds': list(seeds),
                             'whole': [x['whole']['cooperation'] for x in runs],
                             'tail': [x['tail']['cooperation'] for x in runs],
                             'whole_abs_ni': [x['whole']['mean_abs_ni'] for x in runs],
                             'whole_difference_from_full': [x['whole']['cooperation']-y['whole']['cooperation'] for x, y in zip(runs, refs)],
                             'tail_difference_from_full': [x['tail']['cooperation']-y['tail']['cooperation'] for x, y in zip(runs, refs)]})
    return rows


def write_report(root, audit, introduction):
    text = ['# 机制检查：开发验证', '', *introduction, '',
            '| 条件 | 方法 | 全程合作 | 末段合作 | 全程绝对NI | 全程合作减完整模型 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for x in audit['summaries']:
        text.append(f"| r={x['r']}, rho={x['rho']} | {x['name']} | {np.mean(x['whole']):.6f} | "
                    f"{np.mean(x['tail']):.6f} | {np.mean(x['whole_abs_ni']):.8f} | {np.mean(x['whole_difference_from_full']):.6f} |")
    text += ['', '每个条件四个独立环境种子；原始逐次值与配对差保存在 verification.json。',
             '工具来源：沿用 RESEARCH_PLAN.md 中实验设计和统计分析技能的来源记录。']
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    audit['analysis_sha256'] = digest(Path(__file__))
    (root / 'verification.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    (root / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refinement', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--thesis', type=Path, required=True)
    args = parser.parse_args()
    manifest, records = new_runs(args.refinement, 16)
    result = json.loads((args.refinement / 'refinement.json').read_text())
    base_path = Path(manifest['base_path'])
    base = json.loads((base_path / 'calibration.json').read_text())
    base_audit = json.loads((base_path / 'verification.json').read_text())
    assert digest(base_path / 'calibration.json') == result['base_calibration_sha256'] == base_audit['sources']['calibration.json']
    left, right = [next(x for x in base['candidates'] if x['kappa'] == k) for k in [.025, .03]]
    target = base['target_whole_mean_abs_ni']
    kappa = left['kappa'] + (target-left['whole_mean_abs_ni'])/(right['whole_mean_abs_ni']-left['whole_mean_abs_ni'])*(right['kappa']-left['kappa'])
    assert manifest['interpolated_kappa'] == kappa
    assert all(x['config']['method'] == 'cooperation_first' and x['config']['kappa'] == kappa for x in records)
    assert {x['config']['seed'] for x in records} == set(range(104, 108))
    measured = float(np.mean([x['whole']['mean_abs_ni'] for x in records]))
    candidates = [{'kappa': x['kappa'], 'whole_mean_abs_ni': x['whole_mean_abs_ni']} for x in base['candidates']]
    candidates.append({'kappa': kappa, 'whole_mean_abs_ni': measured})
    assert candidates == result['candidates']
    selected = min(candidates, key=lambda x: (abs(x['whole_mean_abs_ni']-target), x['kappa']))
    assert result['selected_kappa'] == selected['kappa'] == kappa
    assert result['target_whole_mean_abs_ni'] == target
    assert result['selected_whole_mean_abs_ni'] == measured
    relative = abs(measured-target)/target
    assert result['pooled_relative_difference'] == relative
    assert result['within_10_percent_pooled'] == (relative <= .1)
    original_manifest = json.loads((base_path / 'manifest.json').read_text())
    full = base['full_model']['label']
    names = ['calibrated_fixed'] * len(records)
    for i, label in enumerate(original_manifest['labels']):
        if label == full:
            rec = read_run(base_path / f'run-{i:04d}.json')
            assert rec['config'] == original_manifest['configs'][i]
            records.append(rec); names.append('full')
    assert len(records) == 32
    amplitude = {'verified_new_runs': 16, 'verified_reused_runs': 16,
                 'selected_kappa': kappa, 'within_10_percent_pooled': relative <= .1,
                 'pooled_relative_difference': relative,
                 'sources': {name: digest(args.refinement / name) for name in ['manifest.json', 'refinement.json']},
                 'summaries': summaries(records, names, 'full', range(104, 108))}
    write_report(args.refinement, amplitude,
                 [f'一次已记录的插值校准；固定 κ={kappa:.17g}，汇总幅度相对差 {relative:.2%}。',
                  '原九点网格未达到容差，本补充不冒充原先预定网格；两阶段均只依据NI幅度选择。',
                  '汇总匹配不等于逐条件匹配，也不控制策略轨迹、来源选择和修正触发频率。'])
    manifest, records = new_runs(args.features, 48)
    names = list(manifest['names'])
    for name, rec in zip(names, records):
        expected = list(manifest['full_model']['controller'])
        for i in manifest['zeroed_indices'][name]: expected[i] = 0.
        assert rec['config']['controller'] == expected
        assert rec['config']['seed'] in range(212, 216)
        assert rec['config']['kappa'] == manifest['full_model']['kappa']
        assert rec['config']['gate_scale'] == manifest['full_model']['gate_scale']
    for item in manifest['reused_baselines']:
        path = Path(item['path'])
        assert digest(path) == item['sha256']
        rec = read_run(path)
        assert rec['config']['controller'] == manifest['full_model']['controller']
        records.append(rec); names.append('full')
    assert len(records) == 64
    features = {'verified_new_runs': 48, 'verified_reused_runs': 16,
                'zeroed_indices': manifest['zeroed_indices'],
                'sources': {'manifest.json': digest(args.features / 'manifest.json')},
                'summaries': summaries(records, names, 'full', range(212, 216))}
    write_report(args.features, features,
                 ['训练后置零，不重新训练，不重新选择控制器。',
                  '这是当前参数对特征的依赖检查，不能证明去掉特征并重训后的最优性能。',
                  '仍计算全部特征；不用于比较节省的执行开销。'])
    dst = args.thesis / 'figures/work2-mechanism'
    dst.mkdir(parents=True, exist_ok=True)
    for stem, audit, caption, names_cn in [
        ('amplitude', amplitude, '汇总修正幅度校准后的开发验证；各条件四次独立运行',
         {'full': '完整模型', 'calibrated_fixed': '幅度校准固定规则'}),
        ('features', features, '冻结参数下的特征移除检查；各条件四次独立运行',
         {'full': '完整模型', 'no_residual': '去掉预测残差', 'no_q_proxy': '去掉Q差距代理', 'neither': '同时去掉两项'})]:
        tex = [r'\begin{table}[!htbp]', r'\centering\small', r'\caption{' + caption + '}',
               r'\label{tab:w2-' + stem + '}', r'\begin{tabular}{cclrrr}', r'\toprule',
               r'$r$ & 异常比例 & 方法 & 全程合作 & 末段合作 & 全程绝对NI \\', r'\midrule']
        for x in audit['summaries']:
            tex.append(f"{x['r']:g} & {x['rho']:g} & {names_cn[x['name']]} & {np.mean(x['whole']):.4f} & "
                       f"{np.mean(x['tail']):.4f} & {np.mean(x['whole_abs_ni']):.5f}" + r' \\')
        tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        (dst / f'{stem}-table.tex').write_text('\n'.join(tex) + '\n')
    provenance = {str(root.resolve()): digest(root / 'verification.json') for root in [args.refinement, args.features]}
    (args.thesis / 'validation/work2-mechanism-sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'refinement_new_runs': 16, 'feature_new_runs': 48, 'selected_kappa': kappa,
                      'relative_ni_difference': relative}))


if __name__ == '__main__':
    main()
