"""Archive checks and descriptive search curves; scores use changing training seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    manifest=json.loads((root/'manifest.json').read_text())
    completion=json.loads((root/'completion.json').read_text())
    for name,digest in manifest['sources'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    report=['# 控制器训练记录','',
            '训练分数来自每代使用的训练环境，不能代替独立验证或留出测试。',
            f'每个候选在四种r/异常条件各运行{manifest["steps"]}步，目标为全程标准化真实福利，严格等于全程合作率。','',
            '| 优化种子 | 代数 | 仿真次数 | 末代候选最高分 | 末代候选平均分 |',
            '| --- | ---: | ---: | ---: | ---: |']
    count=0
    for opt in completion['optimizers_completed']:
        records=[json.loads(f.read_text()) for f in sorted((root/f'optimizer-{opt}').glob('generation-*.json'))]
        assert len(records)==manifest['generations']
        for g,record in enumerate(records):
            assert record['generation']==g
            assert len(record['episodes'])==manifest['population']*4
            scores=np.zeros(manifest['population'])
            for ep in record['episodes']:
                assert 110<=ep['config']['seed']<=199
                assert ep['config']['seed']==record['environment_seed']
                assert ep['config']['steps']==manifest['steps']
                assert ep['config']['controller']==record['candidates'][ep['candidate']]
                assert abs(ep['objective']-ep['whole']['cooperation'])<1e-9
                scores[ep['candidate']]+=ep['objective']/4
                count+=1
            np.testing.assert_allclose(scores,record['scores'],rtol=0,atol=1e-12)
        output=json.loads((root/f'optimizer-{opt}'/'candidates.json').read_text())
        assert output['final_mean']==records[-1]['mean']
        assert output['last_generation_best']==records[-1]['best']
        x=np.arange(1,len(records)+1)
        axes[0].plot(x,[max(r['scores']) for r in records],marker='o',label=str(opt))
        axes[1].plot(x,[np.mean(r['scores']) for r in records],marker='o',label=str(opt))
        last=records[-1]['scores']
        report.append(f'| {opt} | {len(records)} | {len(records)*manifest["population"]*4} | {max(last):.6f} | {np.mean(last):.6f} |')
    assert count==completion['episodes']
    for ax,title in zip(axes,['Best candidate in each generation','Mean of candidate population']):
        ax.set_title(title);ax.set_xlabel('Generation');ax.set_ylabel('Training objective');ax.set_ylim(0,1);ax.legend(title='Optimizer seed')
    fig.suptitle('Development search; training environment seed changes between generations')
    fig.savefig(root/'search.pdf');fig.savefig(root/'search.png',dpi=160);plt.close(fig)
    report+=['','每个优化初始化均保存最终分布均值与末代最佳候选，之后统一进入独立验证。',
             '跨代环境种子变化，不能把曲线波动全部解释为优化退步；候选总体均值也不等于输出控制器的表现。',
             f'已核验{count}次评价、所有代的得分汇总与源码快照。批次墙钟{completion["wall_seconds"]:.1f}秒，包含并行启动与存储，非公平算法耗时。',
             '', '工具来源：Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://doi.org/10.48550/arXiv.2609.00065 。']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report[:12]))

if __name__=='__main__':main()
