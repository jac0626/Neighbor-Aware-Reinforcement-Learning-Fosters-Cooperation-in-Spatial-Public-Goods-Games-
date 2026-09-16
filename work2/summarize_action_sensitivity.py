"""Audit explicit cooperative-indicator removal, reusing the feature audit logic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import summarize_mechanism_checks as audit_helpers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--thesis', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    manifest, records = audit_helpers.new_runs(root, 16)
    assert manifest['zeroed_indices'] == {'no_cooperation_indicator': [4, 9]}
    assert manifest['names'] == ['no_cooperation_indicator'] * 16
    expected = list(manifest['full_model']['controller'])
    expected[4] = expected[9] = 0.
    names = list(manifest['names'])
    for rec in records:
        assert rec['config']['controller'] == expected
        assert rec['config']['seed'] in range(212, 216)
        assert rec['config']['kappa'] == manifest['full_model']['kappa']
        assert rec['config']['gate_scale'] == manifest['full_model']['gate_scale']
    for item in manifest['reused_baselines']:
        path = Path(item['path'])
        assert audit_helpers.digest(path) == item['sha256']
        rec = audit_helpers.read_run(path)
        assert rec['config']['controller'] == manifest['full_model']['controller']
        records.append(rec); names.append('full')
    assert len(records) == 32
    rows = audit_helpers.summaries(records, names, 'full', range(212, 216))
    result = {'verified_new_runs': 16, 'verified_reused_runs': 16,
              'summaries': rows, 'sources': {'manifest.json': audit_helpers.digest(root / 'manifest.json')},
              'action_analyzer_sha256': audit_helpers.digest(Path(__file__))}
    audit_helpers.write_report(root, result,
        ['将已训练参数中的w5和v5置零；不重新训练、不重新选型。',
         '删除显式合作指示量，其他动作信息和NI符号仍保留，不能称为删除全部动作信息。'])
    # Preserve both files needed to repeat this analysis from the archived folder.
    (root / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    (root / 'summarize_mechanism_checks.py').write_bytes(Path(audit_helpers.__file__).read_bytes())
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{冻结控制器中显式合作指示量的移除；每个条件四次开发运行}',
           r'\label{tab:w2-action-feature}', r'\begin{tabular}{cclrrr}', r'\toprule',
           r'$r$ & 异常比例 & 方法 & 全程合作 & 末段合作 & 全程绝对NI \\', r'\midrule']
    for row in rows:
        name = '完整模型' if row['name'] == 'full' else '移除合作指示量'
        tex.append(f"{row['r']:g} & {row['rho']:g} & {name} & {np.mean(row['whole']):.4f} & "
                   f"{np.mean(row['tail']):.4f} & {np.mean(row['whole_abs_ni']):.5f}" + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip\footnotesize 数据来源：本文冻结参数特征移除实验，2026年。', r'\end{table}']
    (args.thesis / 'figures/work2-mechanism/action-table.tex').write_text('\n'.join(tex) + '\n')
    (args.thesis / 'validation/work2-action-sources.json').write_text(json.dumps({str(root.resolve()): hashlib.sha256((root / 'verification.json').read_bytes()).hexdigest()}, indent=2) + '\n')
    print(json.dumps({'verified_new_runs': 16, 'verified_reused_runs': 16}))


if __name__ == '__main__': main()
