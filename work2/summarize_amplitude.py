"""Audit calibration inputs and report cooperation without using it for selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    manifest = json.loads((root / 'manifest.json').read_text())
    calibration = json.loads((root / 'calibration.json').read_text())
    done = json.loads((root / 'completion.json').read_text())
    rows = list(csv.DictReader((root / 'summary.csv').open()))
    assert len(rows) == len(manifest['configs']) == done['completed_runs'] == 112
    assert sorted(int(row['run']) for row in rows) == list(range(112))
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
    records = []
    for row in rows:
        i = int(row['run'])
        rec = json.loads((root / f'run-{i:04d}.json').read_text())
        assert rec['config'] == manifest['configs'][i]
        assert rec['config']['steps'] == 100000
        assert row['label'] == manifest['labels'][i]
        for metric, value in rec['tail'].items():
            assert row[metric] == '' if value is None else np.isclose(float(row[metric]), value, rtol=0, atol=1e-12)
        for column, metric in [('whole_cooperation', 'cooperation'), ('whole_mean_abs_ni', 'mean_abs_ni')]:
            assert float(row[column]) == rec['whole'][metric]
        with np.load(root / f'run-{i:04d}.npz') as data:
            assert hashlib.sha256(data['q'].tobytes()).hexdigest() == rec['q_sha256']
            assert hashlib.sha256(data['bad_mask'].tobytes()).hexdigest() == rec['mask_sha256']
        rec['label'] = row['label']
        rec['source'] = str(root.resolve() / f'run-{i:04d}.json')
        records.append(rec)
    old_manifest = json.loads((root / 'source/direction-manifest.json').read_text())
    old_root = Path(manifest['direction_path'])
    old_files = sorted((root / 'source').glob('direction-run-*.json'))
    assert len(old_files) == 48
    for path in old_files:
        rec = json.loads(path.read_text())
        name = path.name.removeprefix('direction-')
        i = int(name.removeprefix('run-').removesuffix('.json'))
        assert rec['config'] == old_manifest['configs'][i]
        assert path.read_bytes() == (old_root / name).read_bytes()
        with np.load(old_root / name.replace('.json', '.npz')) as data:
            assert hashlib.sha256(data['q'].tobytes()).hexdigest() == rec['q_sha256']
            assert hashlib.sha256(data['bad_mask'].tobytes()).hexdigest() == rec['mask_sha256']
        rec['label'] = f"cooperation_first-{rec['config']['kappa']:g}"
        rec['source'] = str(old_root.resolve() / name)
        records.append(rec)
    full = calibration['full_model']['label']
    table = {(rec['label'], rec['config']['r'], rec['config']['rho'], rec['config']['seed']): rec for rec in records}
    assert len(table) == 160
    target = float(np.mean([rec['whole']['mean_abs_ni'] for rec in records if rec['label'] == full]))
    assert target == calibration['target_whole_mean_abs_ni']
    distances = []
    for candidate in calibration['candidates']:
        label = f"cooperation_first-{candidate['kappa']:g}"
        runs = [rec for rec in records if rec['label'] == label]
        assert len(runs) == 16
        value = float(np.mean([rec['whole']['mean_abs_ni'] for rec in runs]))
        assert value == candidate['whole_mean_abs_ni']
        assert abs(value - target) == candidate['distance']
        distances.append((abs(value - target), candidate['kappa']))
    selected = min(distances)[1]
    assert selected == calibration['selected_kappa']
    assert calibration['within_10_percent_pooled'] == (min(distances)[0] / target <= .1)
    labels = [full] + [f'cooperation_first-{candidate["kappa"]:g}' for candidate in calibration['candidates']]
    report = ['# 固定强度的实际更新幅度校准', '',
              '本批用于开发校准，不是最终留出测试。112次新增运行与48次复用记录均已核验。',
              f'完整模型全程平均绝对NI：{target:.8f}。',
              f'按更新幅度选择的固定κ：{selected:g}；汇总相对差：{calibration["pooled_relative_difference"]:.2%}。',
              f'是否达到预定汇总10%容差：{calibration["within_10_percent_pooled"]}。',
              '固定κ的选择没有使用合作率。以下同时保留所有候选，不据效果继续调整强度。', '',
              '| 方法 | r | 异常比例 | 全程合作均值 | 末段合作均值 | 全程绝对NI | 末段合作范围 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    comparisons = []
    for r in [4.4, 4.8]:
        for rho in [0., .1]:
            for label in labels:
                runs = [table[(label, r, rho, seed)] for seed in range(104, 108)]
                for seed, rec in zip(range(104, 108), runs):
                    assert rec['mask_sha256'] == table[(full, r, rho, seed)]['mask_sha256']
                whole = [rec['whole']['cooperation'] for rec in runs]
                tail = [rec['tail']['cooperation'] for rec in runs]
                amplitude = [rec['whole']['mean_abs_ni'] for rec in runs]
                report.append(f'| {label} | {r:g} | {rho:g} | {np.mean(whole):.6f} | {np.mean(tail):.6f} | '
                              f'{np.mean(amplitude):.8f} | {min(tail):.6f}–{max(tail):.6f} |')
            baseline = f'cooperation_first-{selected:g}'
            differences = [table[(full, r, rho, seed)]['whole']['cooperation'] - table[(baseline, r, rho, seed)]['whole']['cooperation'] for seed in range(104, 108)]
            comparisons.append({'r': r, 'rho': rho, 'seeds': list(range(104, 108)), 'whole_paired_differences': differences})
    report += ['', '汇总幅度匹配不等于每个条件或每个节点逐步匹配；单元幅度差须结合原始记录解释。',
               '参考选择、正优势触发频率与策略轨迹仍可能不同，结果不证明信息真假识别。',
               '工具来源：沿用RESEARCH_PLAN.md中的实验设计与统计分析技能引用。']
    (root / 'REPORT.md').write_text('\n'.join(report) + '\n')
    audit = {'verified_new_runs': 112, 'verified_reused_runs': 48, 'selected_kappa': selected,
             'within_10_percent_pooled': calibration['within_10_percent_pooled'],
             'comparisons': comparisons,
             'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ['manifest.json', 'calibration.json', 'summary.csv']},
             'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    (root / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: audit[key] for key in ['verified_new_runs', 'verified_reused_runs', 'selected_kappa', 'within_10_percent_pooled']}))


if __name__ == '__main__':
    main()
