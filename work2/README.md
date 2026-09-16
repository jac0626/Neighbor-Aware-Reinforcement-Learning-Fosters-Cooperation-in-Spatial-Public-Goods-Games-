# 工作2可复现实验

本目录用于新建的工作2。算法和参数可按证据调整，不追求旧汇报中的数值。研究问题、信息边界、开发/验证/测试划分见 `RESEARCH_PLAN.md`。

## 环境

本机隔离环境在项目根目录 `.venv-thesis/`。依赖固定在 `requirements.lock.txt`。

```bash
python3 -m venv .venv-thesis
.venv-thesis/bin/python -m pip install -r work2/requirements.lock.txt
.venv-thesis/bin/python -m unittest discover -s work2 -v
```

基础检查覆盖：邻域几何、直接博弈收益枚举、极端配置、动作与奖励时序、κ=0退化、IQL消息独立性、同种子重现、异常节点数及福利恒等式。2026-09-15基础8项均通过；加入控制器后，手工特征/门控计算、并列选择、零门控退化及学习控制器重现等检查使总数增至12项；第二轮加入合作语义表达、全D报告退化和合作方向界检查后共15项，全部通过。测试验证这些契约，不证明算法效果或完成论文复现。

## 预实验

从项目根目录运行，输出目录必须是新目录：

```bash
.venv-thesis/bin/python work2/run_pilot.py --out work2/results/pilot-20260915 --workers 96 --steps 20000
.venv-thesis/bin/python work2/summarize_pilot.py work2/results/pilot-20260915
```

每个方法分别使用30×30周期网格、4个r值、2种异常条件、4个开发种子，共224次独立仿真。随机任务顺序固定；各方法共享同种子对应的初始Q、异常节点、以及固定抽样数的动作随机流。每个进程的数值库内部线程为1，独立运行并行。

- `manifest.json`：所有配置、源码SHA256、核心库版本和指标口径。
- `run-*.json`：逐次配置、末窗口与全程指标、掩码/Q表散列、耗时。
- `run-*.npz`：分段均值轨迹、最终动作、Q表、异常掩码。
- `summary.csv`：逐次末20%窗口结果；`REPORT.md`：均值、样本标准差和范围。
- `cooperation.pdf/png`：原始计算结果的图示。
- `completion.json`：全部任务成功返回后的完成记录。

输出位于Git忽略的results目录，当前本地保留。重要结果在最终交付时连同源码版本归档，不依赖仅有Git提交保存数据。原始输出不覆盖、不按好坏删除。

## 当前结果边界

首批2万步开发实验已完成。r=4.8时全局NI κ=0.5的均值约为干净0.945、异常0.127；较低r或不同κ下表现不同。简单上尾过滤在r=4.8受扰条件下约0.821，但不能据此推广到所有参数。结果来自重建环境，不能替代工作1原图；原图的r=3.6动态尚未在本环境中得到对应现象。

10万步延长实验也已完成224次，输出在 `results/pilot-long-20260915/`；最新判断见 `DEVELOPMENT_FINDINGS.md`。最终测试种子1000–1029保留，模型选择不得使用它们。

方法身份：`ni_global`沿用论文数学定义及全局归一化，攻击下比较收到的报告值和自身真实奖励；`ni_local`用每个接收者的局部最大差归一化。`trimmed_local`使用上部次序统计量排除更高报告，并以局部差归一化；边界并列值保留，因此它不保证恰好移除固定数量的邻居。`iql`完全不使用NI消息，但仍采用可信声誉状态。这一可信观测假设是范围边界。

本目录使用已安装的experimental-design/statistical-analysis技能辅助设计，工具来源已在研究方案和报告注明。

## 可训练候选与独立验证

首轮10参数局部控制器已归档；当前实现为补入合作报告特征后的12参数候选，具体定义见 `MODEL_SPEC.md`。两轮完整模型各完成三个优化初始化、3456次训练仿真。训练脚本支持完整学习、最高奖励选择下的仅门控、合作优先选择下的仅门控、以及固定门控下的仅选择。四类结构训练与统一验证均已完成，结果见下文。

```bash
.venv-thesis/bin/python work2/results/cem-20260915/source/train_controller.py --out work2/results/cem-first-round-replay --workers 96 --population 24 --generations 12 --steps 10000
.venv-thesis/bin/python work2/summarize_training.py work2/results/cem-20260915
.venv-thesis/bin/python work2/results/validation-cem-20260915/source/evaluate_controllers.py --training work2/results/cem-20260915 --out work2/results/validation-first-round-replay --steps 100000 --seeds 200 201 202 203 --workers 96
.venv-thesis/bin/python work2/summarize_validation.py work2/results/validation-cem-20260915
```

训练输出保存每代全部候选和逐次评价；验证同时比较每个优化初始化的最终均值、末代最佳参数和六个简单基线，共192个100000步运行。独立验证按四个r/异常单元的全程均值等权选型，末段表现单独报告。`selection.json`中的“selected_candidate”仅指学习候选中的选择，不代表它优于所有基线。

每批保存使用时的源码快照。重跑已归档实验应调用该批 `source/` 下的脚本，以免当前开发版本变化导致定义不同；上面的命令展示当前开发入口，输出目录不能复用已有目录。

首轮独立验证现已全部完成。重要结论与下一步方法决策见 `DEVELOPMENT_FINDINGS.md`，完整对照见 `results/validation-cem-20260915/REPORT.md`。首轮10参数模型没有超过合作消息优先规则，不作为最终方案冻结。

将已核验验证图表同步到论文：

```bash
.venv-thesis/bin/python work2/export_validation_tex.py work2/results/validation-cem-20260915
```

此命令只导出表格和计算图，并记录输入散列；不会把开发数据改称正式测试。

## 第二轮方向与表示检验

方向/强度开发：`results/direction-20260915/`，192次、种子104–107。结果显示κ=0.2的合作优先比κ=0.5更强，方向限制在这四种条件下变化很小。第二轮训练从等效固定强度0.2出发，保留最大强度0.5供门控调整。

```bash
.venv-thesis/bin/python work2/run_direction_pilot.py --out work2/results/direction-replay --workers 96 --steps 100000
.venv-thesis/bin/python work2/train_controller.py --out work2/results/cem-semantic-replay --workers 96 --population 24 --generations 12 --steps 10000 --optimizer-seeds 3100 3101 3102
```

原始第二轮训练目录为`results/cem-semantic-20260915/`。它与首轮的特征、初始参数不同，不能将二者性能差直接归因于单一特征。第二轮验证已完成224次运行，包含κ=0.2合作优先与合作方向限制。所选候选全程目标0.976750，合作优先κ=0.2为0.931122；这是验证选型结果，该句对应开发选型；冻结后的正式测试现已完成，见最终测试一节。

```bash
.venv-thesis/bin/python work2/results/validation-semantic-20260915/source/evaluate_controllers.py --training work2/results/cem-semantic-20260915 --out work2/results/validation-semantic-replay --steps 100000 --seeds 204 205 206 207 --workers 96
.venv-thesis/bin/python work2/summarize_validation.py work2/results/validation-semantic-20260915
.venv-thesis/bin/python work2/export_validation_tex.py work2/results/validation-semantic-20260915 --tag semantic
```

## 固定合作优先选择的门控消融

`cooperation_gate`仅是训练时的参数限制，执行仍调用相同12参数局部控制器：前5个评分权重固定(1,0,0,0,2)，只搜索后7个门控参数。训练结果记录`training_variant`并保留实际执行方法`learned`。三次独立搜索使用相同候选评价预算；与完整模型的逐代环境种子未作配对，不把训练曲线点差当作配对效应。

```bash
.venv-thesis/bin/python work2/train_controller.py --out work2/results/cem-cooperation-gate-replay --method cooperation_gate --workers 96 --population 24 --generations 12 --steps 10000 --optimizer-seeds 3200 3201 3202
```

原始运行目录`results/cem-cooperation-gate-20260915/`，状态以`completion.json`为准。先完成训练再统一验证，以判断动态门控之外是否仍需要学习邻居评分。

该训练与共同验证现已完成，结果目录`results/validation-cooperation-gate-20260915/`。完整模型与固定合作评分/学习门控的全程目标分别0.976763与0.974683；评分学习的收益较小，且r=4.8干净条件下简化模型更高。运行完整性核验和逐条件配对差见`REPORT.md`、`SCORE_ABLATION.md`。

```bash
.venv-thesis/bin/python work2/results/validation-cooperation-gate-20260915/source/evaluate_controllers.py --training work2/results/cem-semantic-20260915 --compare-training work2/results/cem-cooperation-gate-20260915 --out work2/results/validation-cooperation-gate-replay --steps 100000 --seeds 208 209 210 211 --workers 96
.venv-thesis/bin/python work2/summarize_validation.py work2/results/validation-cooperation-gate-20260915
.venv-thesis/bin/python work2/summarize_ablation.py work2/results/validation-cooperation-gate-20260915
```

另外两项结构消融训练也已完成：`results/cem-selection-only-20260915/`使用优化种子3300–3302，固定门控0.4而学习评分；`results/cem-gate-only-20260915/`使用3400–3402，固定最高奖励参考而学习门控。二者各48个并行进程、同样的24×12×四单元10000步预算；效果比较进入下述共同验证。

## 四类结构共同验证

仅评分与仅门控的两批重训均已完成并核验，各3456次仿真，参数约束检查保存在各批`constraint-audit.json`。四类训练输出在新验证种子212–215上共同评价，当前运行入口如下；同名输出目录已存在，重跑应换新目录。

```bash
.venv-thesis/bin/python work2/evaluate_controllers.py --training work2/results/cem-semantic-20260915 --compare-training work2/results/cem-cooperation-gate-20260915 work2/results/cem-selection-only-20260915 work2/results/cem-gate-only-20260915 --out work2/results/validation-structure-20260915 --steps 100000 --seeds 212 213 214 215 --workers 96
```

512次运行已完成，并由`work2/summarize_validation.py`核对逐次指标与原始文件，再由`work2/summarize_ablation.py`生成各类所选候选的配对差和图表。分析脚本支持两类及四类共同验证，已在前一批320次运行上重算并确认原有四个配对比较完全一致。全部候选仍参与报告，每类分别按全程等权目标选型。

## 泛化开发与机制检查

消息、参数、规模和状态条件已按`GENERALIZATION_PROTOCOL.md`完成，共1152次100000步运行，均通过审计。`run_generalization.py`读取共同验证中冻结的完整模型与固定合作评分模型，并加入四个简单基线；使用216–219号开发验证种子。消息和状态扩展通过19项契约检查及76组旧引擎精确回归。消息/状态入口的96/72次短运行只检验实现，不计入研究结果。状态中断后用`resume_generalization.py`保留189次完整输出，仅恢复99次原配置，恢复时使用归档引擎。各批的`generalization.json`及`REPORT.md`保存完整结果与配对差。

## 固定强度校准与最终测试准备

`AMPLITUDE_PROTOCOL.md`定义全程实际NI幅度的汇总匹配规则；`run_amplitude_calibration.py`复用已核验方向批次并新增112次计划长运行。112次100步入口检查已完成，原始结果与55个来源散列核验通过，短运行不会输出校准选择。长运行已完成并审计；后续插值及选择结果见下文。

`FINAL_TEST_PROTOCOL.md`已冻结主环境、十种方法和配对bootstrap分析，1200次最终测试已全部完成并通过审计，八项主要比较与全部逐次分布见`final-report.json`和`REPORT.md`。结果目录为`results/final-test-20260915`，完成状态以其中`completion.json`为准；不能把目录创建或部分输出称为测试完成。

幅度网格完成112次新增运行、复用48次旧记录，初始九点网格未达到10%容差；按已记录补充协议仅作一次插值，新增16次运行后，固定强度0.027704459701337496与完整模型的汇总NI幅度相差4.40%。原网格记录保留，选择不依据合作率。冻结参数的特征移除完成48次新增运行、复用完整模型16次。核验与论文表格导出：

```bash
.venv-thesis/bin/python work2/summarize_mechanism_checks.py --refinement work2/results/amplitude-refinement-20260915 --features work2/results/feature-sensitivity-20260915 --thesis thesis
```

60次单进程计时已按`TIMING_PROTOCOL.md`完成并由`summarize_timing.py`审计，在L=30和100、相同10000步预算下记录全部十种方法的三次独立运行。共享机器上的墙钟及CPU时间共同报告，计时结果不参与重新选型。计时和最终测试的模型均来源于`validation-structure-20260915`、`amplitude-calibration-20260915`与`amplitude-refinement-20260915`，源码快照随每批保存。

`run_final_test.py`要求四组泛化和幅度校准审计完成后再启动留出测试。`summarize_final_test.py`核对原始记录与轨迹，并通过`paired_analysis.py`计算预定的八个主要比较；目前仅完成实现及统计契约检查，尚无最终测试结果。

显式合作指示量已按`ACTION_SENSITIVITY_PROTOCOL.md`完成冻结参数移除，共16次新增运行，复用结构验证16次完整模型记录，均通过审计。该批不重训、不重新选型，结果见`results/feature-action-sensitivity-20260915/`。

## 最终结果与论文输出

冻结测试完成1200次运行，主指标全程合作相对已调固定合作规则提高4.105–5.255个百分点；学习评分相对固定评分门控的增量较小且随条件改变。主结果、全部30次值和八项区间见`results/final-test-20260915/REPORT.md`及`final-report.json`。`export_final_tex.py`导出主表、轨迹和预定快照，`export_final_mechanism.py`导出预定次要指标的描述性表；没有追加显著性检验。论文来源排版按REPRODUCE.md最后一步执行。
