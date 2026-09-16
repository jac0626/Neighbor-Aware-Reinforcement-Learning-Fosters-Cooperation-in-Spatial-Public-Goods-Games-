# 参考文献核验记录

核验日期：2026-09-15。引用内容与出版元数据分别核对；没有读到的论文内容不据题目推断。

| BibTeX键 | 核验来源 | 在正文中的用途与限制 |
| --- | --- | --- |
| kang2025neighbor | [Crossref登记](https://api.crossref.org/works/10.1016/j.chaos.2025.116862)；作者提供的源稿和原图 | 工作1模型与结果来源；源稿与本地正文相同，出版商最终排版版尚未逐字核对。具体图文问题见work1-audit.md。 |
| sutton2018reinforcement | [MIT Press第二版书目](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) | 强化学习基本概念；核对作者、版本、年份与ISBN，不据出版社简介推导特殊收敛结论。 |
| watkins1992q | [出版DOI](https://doi.org/10.1007/BF00992698)；[作者PDF](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf) | Q-learning及其条件性收敛结论；本文多智能体、压缩状态与固定步长不直接满足原定理全部条件。 |
| yu2024admac | [AAAI正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/29708)；[全文](https://ojs.aaai.org/index.php/AAAI/article/download/29708/31215) | 消息可靠性估计与影响调节的直接相关研究。不能将“消息评分+门控”这一一般思路宣称为本文首创。 |
| deboer2005tutorial | [作者正式论文PDF](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf)；[大学书目](https://research.utwente.nl/en/publications/a-tutorial-on-the-cross-entropy-method/) | 随机候选分布与精英更新的算法来源；本文搜索预算、平滑系数与标准差下限是本项目开发配置，不宣称来源论文证明这些具体配置最优。 |
| brandt2003punishment | [作者正式论文PDF](https://personal.math.ubc.ca/~hauert/publications/reprints/brandt_procb03.pdf) | 空间公共物品博弈中的惩罚与声誉；原文声誉围绕惩罚意愿，不等同于本文按合作动作累积的声誉。 |
| xue2022misspoke | [AAMAS正式全文](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1418.pdf)；[作者预印本摘要](https://arxiv.org/abs/2108.03803) | 消息生成攻击、重构防御与攻防博弈的相关研究，本文未复现其完整训练框架。 |
| macy2002learning | [PNAS开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC128590/)；DOI 10.1073/pnas.092080099 | 期望水平与Bush–Mosteller学习；对象是两人社会困境，不写成Q-learning或空间公共物品博弈。 |
| zheng2024evolution | [出版社页面](https://www.sciencedirect.com/science/article/pii/S0960077924011202)；[作者全文](https://arxiv.org/html/2407.19851v1) | 用一阶邻居C/D人数关系构造三个状态。模型第2节明确只用自己发起的一组奖励，不能与本文五组累计收益阈值直接比较。出版元数据为2024年188卷115568；HTML动态显示日期不替代出版年。 |
| tamura2024analysing | [Royal Society开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11631413/) | DQL输入为自身上一动作与其他成员上一轮合作比例；改变博弈群组规模，不等同于固定五人交互下扩大感知范围。文中是计算机仿真，没有新的人体实验。 |
| li2025otherregarding | [APS正式页面](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.014304)；[作者全文](https://arxiv.org/html/2410.10921v1) | 超图、他人行为历史及反协调模式；只据已读内容作机制定位，不把它作为本文合作团簇的直接证据。最后一名作者采用出版页Li Chen，年份为2025。 |
| gupta2017cooperative | [Springer正式页面](https://link.springer.com/chapter/10.1007/978-3-319-71682-4_5)；[作者保存的正式全文](https://rejuvyesh.com/publications/2017-coop-mas-drl.pdf) | 第4.3节参数共享与分散执行；本文共享的是修正控制器，非该论文的动作策略网络。采用18页正式版元数据，早期8页workshop稿不混作最终版本。 |
| jiang2018attentional | [NeurIPS页面](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html)；[全文](https://proceedings.neurips.cc/paper/2018/file/6a8018b3a00b69c008601b8becae392b-Paper.pdf) | ATOC组织通信群组并融合内部表示；本文只选择已接收消息并缩放NI，不能宣称零门控减少已经发生的通信。 |
| agarwal2021statistical | [NeurIPS页面](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)；[全文](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)；[作者页面](https://agarwl.github.io/rliable/) | 少量RL重复下的聚合估计、区间与分布报告；不把其深度RL基准结论直接用于证明本文置信区间覆盖率，也不机械替换预先选定的任务目标。 |
| perc2017statistical | [作者保存的正式全文](https://www.matjazperc.com/publications/PhysicsReports_687_1.pdf)；[出版商页面](https://www.sciencedirect.com/science/article/pii/S0370157317301424) | 第3.1节空间公共物品模型，第5节及第9节的有限规模讨论。用于研究背景和规模检验动机，不将其特定模仿规则下的合作阈值套用于本文Q-learning。 |
| ma2025decentralization | [作者全文](https://arxiv.org/html/2504.21278v1)；[arXiv书目](https://arxiv.org/abs/2504.21278)；[作者大学页面](https://people.ucas.ac.cn/~lishoubin) | 第3节DMAC通过学习对手屏蔽关键通信通道、重新训练通信策略。与本文固定广播下的消息内容篡改区分。当前按已核实的2025年预印本引用，不依据匿名代码仓库名称填成IJCAI正式论文。 |

| mitchell2020gaussian | [作者全文](https://arxiv.org/pdf/2012.00508)；[arXiv元数据](https://arxiv.org/abs/2012.00508) | 第IV节空间GP与三类消息来源假设，第VI节置信度加权。用于说明概率可信度需要生成假设；不把本文经验残差称为该方法的后验置信度。按核验到的2020年预印本引用。 |
| sun2022certifiably | [已读取的2022年作者稿](https://arxiv.org/pdf/2206.10158)；[arXiv元数据](https://arxiv.org/abs/2206.10158) | 消息子集消融、动作集成及条件性证书，见第4节、附录A/B。定理有攻击预算、采样和投票等条件，不写成任意通信攻击下均能保持动作。ICLR 2023索引可检索到题名调整后的正式论文，但OpenReview正式PDF本次返回403；当前明确引用已读2022年作者稿，后续宜核对正式版后统一书目。 |

| zhang2021multiagent | [Springer正式章节](https://link.springer.com/chapter/10.1007/978-3-030-60990-0_12)；[作者校样全文](https://arxiv.org/pdf/1911.10635) | 核验2021年、SSDC325卷、321–384页与DOI；已读任务分类和第3.2节非平稳性。用于区分共同奖励任务、个体奖励学习及信息结构，不据综述赋予本文新的收敛保证。 |
| salimans2017evolution | [arXiv元数据](https://arxiv.org/abs/1703.03864)；[作者全文](https://arxiv.org/pdf/1703.03864) | 黑盒回报评价与策略参数搜索的相关途径；本文使用CEM精英更新、优化NI控制器，未复现原文ES梯度估计，也不引入其任务速度数字。按已核验2017年预印本引用。 |

| ng1999policy | [作者全文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)；[Berkeley作者出版目录](https://www2.eecs.berkeley.edu/Pubs/Faculty/russell.html) | 核验ICML1999、278–287页、Morgan Kaufmann；已读引言、MDP定义和策略不变性讨论。用于区分附加值更新的有效奖励解释与具有策略不变条件的奖励塑形，不把单智能体结论套为本文多智能体系统的收敛保证。NI与有效奖励增量的代数等价由本文更新式直接推出。 |

当前21条文献分别覆盖研究环境、价值学习、通信与消息攻击、搜索和统计报告。检索到但未核实正文的条目不直接并入参考文献；相关研究的充分性按各项论断是否有直接证据评估，不以数量代替论证。

实验辅助技能的工具来源已列入工作2研究方案和报告：[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)，作者Kassis、Agarwal、He、Patel与Brueckner，2026年。当前记录为2026-09-02修订的v2；引用使用不带版本后缀的DOI。
