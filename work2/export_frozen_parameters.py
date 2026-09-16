"""Export the exact development-selected models without running held-out seeds."""
import argparse
import hashlib
import json
from pathlib import Path

from run_final_test import frozen_models


def main():
    parser = argparse.ArgumentParser()
    for key in ['validation', 'calibration', 'refinement', 'thesis']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    models = frozen_models(args.validation, args.calibration, args.refinement)
    parameters = [f'$w_{i}$' for i in range(1, 6)] + [f'$v_{i}$' for i in range(1, 6)] + ['$v_u$', '$b$']
    tex = [r'\begin{table}[!htbp]', r'\centering\small',
           r'\caption{共同验证选定的控制器参数；展示值保留六位小数}', r'\label{tab:app-frozen-parameters}',
           r'\begin{tabular}{lrrrr}', r'\toprule',
           r'参数 & 完整模型 & 固定评分/门控 & 学习评分/固定门控 & 最高奖励/门控 \\', r'\midrule']
    for i, name in enumerate(parameters):
        tex.append(name + ' & ' + ' & '.join(f"{model['controller'][i]:.6f}" for model in models[:4]) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path = args.thesis / 'figures/work2-frozen-parameters.tex'
    path.write_text('\n'.join(tex) + '\n')
    provenance = {'models': models, 'source_files': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [args.validation / 'manifest.json', args.validation / 'selection.json',
                             args.calibration / 'calibration.json', args.refinement / 'refinement.json']},
                  'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.thesis / 'validation/work2-frozen-parameters.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'exported_models': len(models), 'held_out_runs_executed': 0}))


if __name__ == '__main__': main()
