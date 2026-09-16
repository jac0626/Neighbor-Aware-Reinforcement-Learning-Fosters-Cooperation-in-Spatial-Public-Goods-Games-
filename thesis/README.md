# 云南大学硕士学位论文工程

当前源码已完成一轮论文论证与表达修订，工作1原图、工作2模型、开发检验及1200次最终留出测试均保持不变。2026-09-15核验过的上一编译快照为107页；2026-09-16完成正文与摘要修订后，当前源码尚未重新执行完整 `latexmk main.tex` / `latexmk anonymous.tex`，因此 `build/` 中PDF、源码ZIP及对应编译散列暂视为上一已验证快照，不应当作当前源码的最新构建结果。研究方向、提交年月及新版扉页仍有待定项，当前文件用于导师评阅。

## 研究范围与当前材料

题目：面向局部社会信息的多智能体强化学习与合作演化研究。

工作1保持已出版的Neighbor-aware reinforcement learning研究；工作2按作者授权重新设计，研究不可靠邻居消息下的参考选择与价值修正控制。中期汇报仅提供动机，早期双脑架构与无原始数据的汇报数字均未作为研究结果。

- `build/main.pdf`：上一已验证完整写作稿；当前源码修订后需重新生成。
- `build/anonymous.pdf`：上一已验证匿名写作预览；当前源码修订后需重新生成并复核匿名字段。
- `build/work1-preview.pdf`：17页工作1单章预览，含文献，未随本轮全文文字修订重新生成。
- `build/thesis-source.zip`：上一已验证论文工程快照；当前分支源码更新后需重新打包并刷新`build/source-package-verification.json`。
- `PROGRESS.md`：当前验收进度；实验状态以各批`completion.json`和审计文件为准。

## 已确认信息与待定字段

封面唯一编辑入口为`metadata.tex`。姓名蒋超，学号12024219002，专业软件工程，导师康洪炜副研究员。学院按公开通知采用“软件学院 人工智能学院”；专业学位类型沿用开题材料。

提交年月由作者明确保留待定，研究方向尚待确定，分类号按机器学习内容选用TP181。当前仅列已知导师。不得以编译日期或虚构的评阅、答辩信息填补空项。

## 编译

在本目录执行：

```bash
export PATH=/opt/texlive/2026/bin/x86_64-linux:$PATH
latexmk main.tex
latexmk anonymous.tex
```

两个入口按顺序编译，使用同一套章节源码。`latexmkrc`配置XeLaTeX、BibTeX及`build/`输出目录；本机使用TeX Live 2026。Overleaf选择XeLaTeX及较新TeX Live，将`main.tex`或`anonymous.tex`设为主文件。

最小TeX Live所需包可通过下列命令安装，依赖由tlmgr解析：

```bash
tlmgr install xetex latexmk ctex xecjk fandol fontspec unicode-math geometry fancyhdr footmisc bigfoot ntheorem caption adjustbox tabularray xpatch xits tex-gyre gbt7714 pgf booktabs
```

## 文件结构与证据

`chapters/`包含六章正文，`frontmatter/`包含摘要与符号表，`backmatter/`包含附录、成果与致谢。`setup.tex`保存本项目的排版适配；`references.bib`当前21条，核验来源记录在`validation/reference-sources.md`。

工作1压缩包的19个原始图片文件保持不变，18个图文件按11组结果图接入，另绘制模型示意图。文件散列、图文解释及旧脚本差异见`validation/work1-figure-sources.json`和`validation/work1-audit.md`。缺少原论文逐次运行数据和确切稳态窗口时，不补造误差条或数值复现结论。

工作2代码、协议、原始数据和各批源码快照位于项目`work2/`。基线、四类模型同预算训练、512次共同结构验证、1152次泛化、实际更新幅度校准及特征移除已完成。1200次留出测试已按`work2/FINAL_TEST_PROTOCOL.md`完成并审计，60次公平开销按`work2/TIMING_PROTOCOL.md`完成并通过核验。论文图表与相应数据的散列对应保存在`validation/work2-*-sources.json`。本轮只调整论文论证与表达，没有改动实验代码、原始数据或这些结果图表。

## 模板来源与格式边界

上游：[Astro-Lee/YNUthesis](https://github.com/Astro-Lee/YNUthesis)，固定版本`0842a7fe974de05bd2110db267e7a077a0e20075`，取得日期2026-09-15。上游类文件、定义、标志、文献样式、dtx源文件和许可证保持不变；本地适配写在`setup.tex`。来源和许可证见`README.template.md`、`LICENSE.template`。

本地适配使封面使用已配置日期、显示专业学位标识，并在Fandol环境下使用楷体；页眉、页脚、字体嵌入和匿名字段的既有核验均对应2026-09-15的上一编译快照。当前源码重新编译后，需要再次检查Times New Roman/Fandol字体、页眉页脚、图表浮动、引用、匿名封面和PDF元数据。新版扉页及简况表也需取得学校原表核对。

依据：[云南大学基础写作规范](https://office.ynu.edu.cn/info/1080/1701.htm)、[学院2026年秋季学位授予通知](https://www.sei.ynu.edu.cn/info/1057/2662.htm)。用户毕业批次未定，通知截止日期不直接作为用户提交日期。具体对应与未决项见`validation/format-audit.md`。

上一编译快照的完整稿与匿名预览均曾通过编译，日志未发现未定义引用、缺字或内容溢出，并完成匿名封面目视检查；这些结果不能自动继承到2026-09-16后的文字修订源码。当前源码在正式送审或重新打包前必须重新完整编译，并刷新PDF、日志、页数和源码包散列。