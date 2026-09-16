# 写作与实验技能：安装记录及验证

> 本文件是首次安装时的验证存档。下文汇报数字仅用于当时的表达与算术检查，未作为工作2实验结果。作者随后确认工作2无原始数据并授权重建；当前研究状态见`../PROGRESS.md`及项目`work2/REPRODUCE.md`。

日期：2026-09-15。安装位置为本项目 `.agents/skills/`，未修改全局技能目录。使用系统 skill-installer 的 GitHub 安装脚本，固定上游提交。

| 技能 | 上游 | 固定提交 |
| --- | --- | --- |
| research-writing-skill | [Chen-ShiRui/claude-academic-skills](https://github.com/Chen-ShiRui/claude-academic-skills/tree/7ed6377f0efb6a38951b48ef03b19d996e454b1f/research-writing-skill) | `7ed6377f0efb6a38951b48ef03b19d996e454b1f` |
| experimental-design | [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/330c8e764435a731eff571e3efdda70b363d0792/skills/experimental-design) | `330c8e764435a731eff571e3efdda70b363d0792` |
| statistical-analysis | [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/330c8e764435a731eff571e3efdda70b363d0792/skills/statistical-analysis) | 同上 |

本地兼容修改：experimental-design 的 YAML `compatibility` 从根字段移入 `metadata`，保留原值，以通过当前系统 quick_validate.py 校验。三个技能均通过入口结构校验。项目写作边界记录在 `thesis/AGENTS.md`。后续更新技能时保留该兼容修改，或以届时校验器支持情况为准。

各技能目录的 `LICENSE.upstream` 保留对应固定提交的上游根目录许可证。

## 验证产物

- `../build/skill-validation.pdf`：工作1正文样例、工作2实验审查与来源说明。
- `work1-sample.tex`：可继续修改的正文样例，尚未合入正式章节。
- `check_claims.py`：对汇报中四个均值的差值和有条件的福利恒等式进行计算。
- `../build/skill-validation-checks.json`：实际执行输出。3×3周期格点、两个 r 值，共1024个策略配置的精确算术检查。

正文事实依据：`work1-paper/en_main.tex` 的 Basic Setup、Q-Learning 和 Neighbor Influence 小节。这里只核对了用户提供的本地稿件，未再次逐字核对出版商最终排版版。工作2数字来自会话中的中期汇报，当前文件检索未定位并核验其逐次实验数据。

## 本次发现

1. 工作1参考邻居时比较综合奖励，不能统一替换为原始收益。它还包含全局差值归一化，不宜描述为完全局部执行。
2. r=4.8 时异常条件的汇报均值差为0.640，无故障性能代价为0.186。尚不能计算置信区间或声称统计显著；0.005的跨故障差值不足以说明故障有利。
3. 固定 r、每人参与五组、每组成本1时，真实平均原始收益等于 `5(r-1)f_C`。这个数学关系须结合工作2的真实目标函数解释，不能仅凭目标里没有出现合作率，就认定二者无关。
4. 后续优先核验逐次结果和目标函数；再以同设置下的小固定修正强度、仅学习选择、仅学习门控及完整模型区分机制。独立单位是运行，不是节点或时间步。提前停止与未收敛运行都要在结果中交代。

## 如何复验

从项目根目录运行：

```bash
python3 thesis/validation/check_claims.py
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/research-writing-skill
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/experimental-design
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/statistical-analysis
```

从 `thesis/` 目录编译：

```bash
PATH=/opt/texlive/2026/bin/x86_64-linux:$PATH latexmk skill-validation.tex
```

本轮通过直接读取技能进行了应用。新安装技能按安装器说明在下一轮可用；若界面仍未列出，重启 Codex 后在本项目调用。可直接要求“用 research-writing-skill 撰写第三章”或“用 experimental-design 和 statistical-analysis 检查实验”。

本次没有对“加载与不加载技能”进行独立盲评，不能量化写作提升，也不承诺所谓检测通过率。提供样稿供用户实际判断。没有安装或验证全部可选数值依赖，也没有运行训练；技能自带随机化、DOE和假设检查脚本不属于本轮已验证范围。

## 工具引用

依照 K-Dense 两项技能的引用要求，本报告记录：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。2026-09-15核对 arXiv 记录，当前修订日期为2026-09-02。该条是本验证报告的工具来源说明，尚未加入学位论文参考文献。
