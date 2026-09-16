"""Descriptive development report: no significance claims or final-test selection."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    rows=list(csv.DictReader((root/'summary.csv').open()))
    manifest=json.loads((root/'manifest.json').read_text())
    if len(rows)!=len(manifest['configs']): raise ValueError('Incomplete pilot')
    groups=defaultdict(list)
    for row in rows:
        groups[(row['method'],float(row['kappa']),float(row['r']),float(row['rho']))].append(row)
    variants=list(dict.fromkeys((c['method'],c['kappa']) for c in manifest['configs']))
    rs=sorted({float(row['r']) for row in rows})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.9),sharey=True,layout='constrained')
    report=['# 工作2基线预实验：描述性结果','',
            '这是开发种子100–103的预实验，每个条件4次独立运行。',
            '尚无学习控制器；结果不等于复现已发表图或旧汇报数值，也不属于最终测试。',
            f"每次运行{rows[0]['steps']}步，合作率按末20%时间窗口平均。无提前停止。",'',
            '| 方法 | κ | r | 异常比例 | 合作率均值 | 样本标准差 | 最小–最大 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for ax,rho in zip(axes,[0.,.1]):
        matrix=np.empty((len(variants),len(rs)))
        for i,(method,kappa) in enumerate(variants):
            for j,r in enumerate(rs):
                cell=groups[(method,kappa,r,rho)]
                values=np.array([float(row['cooperation']) for row in cell])
                matrix[i,j]=values.mean()
                report.append(f'| {method} | {kappa:g} | {r:g} | {rho:g} | {values.mean():.4f} | {values.std(ddof=1):.4f} | {values.min():.4f}–{values.max():.4f} |')
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        for i in range(len(variants)):
            for j in range(len(rs)):
                ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',
                        color='black' if matrix[i,j]>.6 else 'white',fontsize=9)
        ax.set_xticks(range(len(rs)),[f'{r:g}' for r in rs]);ax.set_xlabel('Synergy factor r')
        ax.set_yticks(range(len(variants)),[f'{method}, k={k:g}' for method,k in variants])
        ax.set_title('Clean messages' if rho==0 else '10% persistent high-reward/defection senders',fontsize=10)
    fig.colorbar(im,ax=axes,label='Mean tail cooperation (4 development seeds)',shrink=.85)
    fig.suptitle('Development pilot; 30 x 30 lattice; no learned controller',fontsize=12)
    fig.savefig(root/'cooperation.pdf');fig.savefig(root/'cooperation.png',dpi=180);plt.close(fig)
    completion=json.loads((root/'completion.json').read_text())
    report += ['', '## 运行和解释边界','',
               f"完整运行数：{len(rows)}；并行进程：{manifest['workers']}；批次墙钟时间：{completion['wall_seconds']:.1f}秒。",
               '任务之间并行且本机存在其他负载，进程耗时不能直接当作公平的算法性能基准。',
               '原始JSON保留每个运行的配置和诊断量，NPZ保存分段轨迹、最终动作、Q表和异常节点掩码。',
               'summary.csv为逐次运行的末窗口汇总；此处标准差衡量运行间差异，不是误差条或置信区间。',
               'ni_global按收到的报告和自身真实奖励计算全局差值；ni_local使用接收者局部最大差值。',
               'trimmed_local排除严格高于上部次序统计量的消息；当多个最高值并列时可能保留这些并列消息，并非固定删除恰好20%的节点。',
               '异常被选比例需区分所有候选选择与正优势更新事件，不能将两种分母混用。',
               '后续模型选择只使用开发/验证集。若采用预实验选择参数，正文将其列为开发过程而不是独立验证。',
               '', '## 工具来源','',
               '实验设计参考已安装的K-Dense技能：Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://doi.org/10.48550/arXiv.2609.00065 。']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    for r in rs:
        print('r=',r)
        for method,kappa in variants:
            means=[np.mean([float(row['cooperation']) for row in groups[(method,kappa,r,rho)]]) for rho in [0.,.1]]
            print(f'  {method} k={kappa:g}: clean={means[0]:.4f} fault={means[1]:.4f}')

if __name__=='__main__': main()
