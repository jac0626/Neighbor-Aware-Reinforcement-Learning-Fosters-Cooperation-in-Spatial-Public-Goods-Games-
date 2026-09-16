# 实验复现与图表追溯

## 先确定要复现的对象

第三章的原图来自已发表工作，原论文逐次数据未提供，不能由本工程重建其置信区间。新增程序明确采用同轮动作与收益时序，与部分保存的旧脚本不同。工作2的结果以各批源码快照、配置和逐次输出为依据。

首轮10参数模型与之后12参数模型使用不同引擎；复现历史结果时读取对应批次`source/engine.py`，不要默认用当前引擎替换。每个批次的`manifest.json`记录配置和源码SHA256，`completion.json`记录是否完整结束。恢复过的状态批次额外保存`interruption.json`及恢复清单。

## 环境

当前实测环境为Linux、Python 3.10、NumPy 2.2.6、Numba 0.67.0、SciPy 1.15.3、Matplotlib 3.10.9。完整固定依赖见`requirements.lock.txt`。建议新建虚拟环境后安装这些版本，避免改变已有环境。当前机器的可用环境位于项目根目录`.venv-thesis/`。

## 重放一条记录

在项目根目录执行，例如重放已经完成的规模开发第0项：

```bash
.venv-thesis/bin/python work2/replay_run.py work2/results/generalization-scale-20260915 --run 0 --out work2/results/replayed-scale-run0
```

脚本核验并加载该批归档引擎，从清单恢复原配置与原种子，保存新的数组和比较报告。输出目录必须不存在，原始结果保持不变。比较Q表、动作、异常掩码、分段轨迹及全程/末段汇总；跨硬件或依赖版本的浮点一致性须由实际比较确认。该命令执行完整100000步，耗时取决于规模和方法。

训练候选的重跑使用对应`source/train_controller.py`及原命令参数，训练初始化、种子池、代数、种群与环境条件见训练清单和README。不要把末代最佳候选与最终分布均值当作两次独立训练。

## 图表对应

| 论文证据 | 数据目录（均在results/下） | 分析入口或记录 |
| --- | --- | --- |
| 初始10参数验证 | validation-cem-20260915 | summarize_validation.py；work2-development-sources.json |
| 12参数验证 | validation-semantic-20260915 | summarize_validation.py；work2-semantic-sources.json |
| 固定评分重训比较 | validation-cooperation-gate-20260915 | summarize_ablation.py |
| 四类结构共同验证 | validation-structure-20260915 | summarize_ablation.py；selection.json |
| 消息、参数、状态、规模开发 | generalization-{messages,parameters,state,scale}-20260915 | summarize_generalization.py；export_generalization_tex.py |
| 固定规则幅度网格 | amplitude-calibration-20260915 | summarize_amplitude.py；calibration.json |
| 一次插值补充 | amplitude-refinement-20260915 | summarize_mechanism_checks.py；refinement.json |
| 冻结参数的特征移除 | feature-sensitivity-20260915 | summarize_mechanism_checks.py |
| 显式合作指示量移除 | feature-action-sensitivity-20260915 | ACTION_SENSITIVITY_PROTOCOL.md；summarize_action_sensitivity.py |
| 单进程开销 | timing-20260915 | summarize_timing.py；timing.json；逐次墙钟及CPU时间 |
| 最终留出测试 | final-test-20260915 | FINAL_TEST_PROTOCOL.md；summarize_final_test.py |

各批完成状态须查看`completion.json`，不能将计划或部分输出作为完整结果。论文的`validation/`目录保存导出图表的来源散列。绘图采用标准Matplotlib与真实记录，未生成或补造实验数值。

## 校准与统计

幅度初始九点网格没有达到10%容差；追加一次已记录的插值后固定κ=0.027704459701337496。两个阶段只按全程绝对NI选择，旧候选和未达到容差的结果均保留。特征移除是在固定参数上置零，未重训。

正式测试的模型按共同结构验证选型；最终种子1000–1029仅用于冻结后的测试。八项主要配对差采用同一组环境种子块进行100000次bootstrap，同时报告95%和99.375%百分位区间。完整逐次值保留，节点和时间步不计作独立重复。统计细节以测试前冻结协议为准。

部分开发清单保存了原机器的绝对来源路径，用于记录复用关系。搬迁目录时，单条重放不依赖这些路径；若重新执行跨批汇总脚本，应使相应来源目录可访问，或在单独工作副本中明确重定位路径并保留原始清单。不要为方便重跑改写原归档文件的内容或散列。

论文图表导出后，执行 `.venv-thesis/bin/python thesis/validation/format_figure_sources.py`，将来源标注统一置于图表下方。该排版步骤不改变数值行、图像或原始实验记录；导出散列与排版后散列分别保留，转换记录见`thesis/validation/figure-source-formatting.json`。
