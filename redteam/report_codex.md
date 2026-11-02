已逐行审完这 13 个文件，未修改文件、未调用模型。复算了现存结果、阶梯和事件脚本，并运行了内存反例；没有重跑完整 Fashion-MNIST 训练，也没有给仍在运行的 MAS 实验做最终评分。

最硬的两个结果：**单秤归因可达到 100%；按通常的相对误差 >20% 评分，约束 AUC 是 0.690，未达到 0.70。**

1. **单秤“分不清”的结果由打分丢掉符号造成。**  
   **类型：**假通过、基线不公平。**严重度：严重。**  
   **证据：**[ladder2.py:42](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/ladder2.py:42) 对变化取绝对值，[ladder2.py:121](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/ladder2.py:121) 只用这个绝对分数归因；但 [ladder2.py:103](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/ladder2.py:103) 和第 105 行把节点、秤的故障都限制为正向注入。于是“共识−A”的变化在两类里方向相反。实测只保留这个符号，SLQA、BIPM 的全部 δ 下归因均为 **100%**，原报告为 44–50%。这使当前实验无法支持单秤不可归因；T3 关于任意偏移的命题需要另外检验。  
   **判决检验：**在相同数据上用 `median(ev共识−A)−median(cal共识−A)` 的正负分类，并进一步让注入方向未知。

2. **“错超过 20%”的实现改变了 AUC 是否通过。**  
   **类型：**假通过、代码缺陷。**严重度：严重。**  
   **证据：**[constraint.py:66](/Users/sly/Desktop/agent/claude/ring-arith-toy/llm/constraint.py:66)、[constraint.py:75](/Users/sly/Desktop/agent/claude/ring-arith-toy/llm/constraint.py:75) 使用 `abs(log10(预测/真值)) > log10(1.2)`。这把低估超过 **16.67%** 也算成“错超过 20%”。改成通常的 `abs(预测/真值−1)>0.2`，错误样本从 **83 降至 72**，约束 AUC 从 **0.719752 降至 0.689781**，低于预注册门槛。  
   **判决检验：**保持信号和样本不变，仅按文档中的相对误差 >20% 重算标签和 AUC。

3. **SLQA 输入包含不能抵消的逐实验室真值。**  
   **类型：**数据泄漏、隐含假设。**严重度：严重。**  
   **证据：**[slqa.py:17](/Users/sly/Desktop/agent/claude/ring-arith-toy/slqa/slqa.py:17) 输入为 `log(rep/act)`；[ladder2.py:68](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/ladder2.py:68) 直接使用这些输入。每个实验室的瓶子真值不同，比较时仍包含 `log(act_j/act_i)`。检查的 **207 个题目全部存在实验室间真值差异**，所以这里不能用“共同减去真值会抵消”解释。原始报告比较的内存消融仍得到阶梯 5/5，因此泄漏对通过判定的贡献尚未证实。  
   **判决检验：**禁止检测路径读取逐瓶 `act`，只允许原始报告和事先可得的信息，重算阶梯及 distinguish。

4. **三角帽的方差恒等式被错误套在 MAD² 上。**  
   **类型：**数学错误、假失败。**严重度：严重。**  
   **证据：**[premise2.py:29](/Users/sly/Desktop/agent/claude/ring-arith-toy/switching/premise2.py:29)、[premise2.py:33](/Users/sly/Desktop/agent/claude/ring-arith-toy/switching/premise2.py:33) 把三项 MAD² 当方差相加减；MAD² 一般没有方差的可加性。已运行反例：9 个独立节点噪声标准差 0.001，两把独立秤均为 `Uniform(-1,1)`，4000 题、种子 123。真实共识标准差约 **0.000405**，代码却估计共有误差 **0.415**，并在 `h=0.1` 时拒答。  
   **判决检验：**在全部来源独立的高斯、均匀和厚尾噪声下测名义误拒率，检查 N1 是否仍把分布形状误认为共有误差。

5. **约束实验重新使用了仓库自己已否决的真值。**  
   **类型：**假失败、代码缺陷。**严重度：严重。**  
   **证据：**[证据链总结.md:105](/Users/sly/Desktop/agent/claude/ring-arith-toy/证据链总结.md:105) 已指出多米尼加、加拿大、克罗地亚面积数据错误；[constraint.py:29](/Users/sly/Desktop/agent/claude/ring-arith-toy/llm/constraint.py:29)、[constraint.py:69](/Users/sly/Desktop/agent/claude/ring-arith-toy/llm/constraint.py:69) 又直接拿同一份数据产生标签。这三国贡献 **19/83** 个错误标签。仅删除三国的敏感性检查就让 AUC 变成 **0.814**，说明“自洽地错”的比例也受到标签污染。  
   **判决检验：**独立核验这三国的评分真值，固定修正规则后同时重算 AUC、错误率和自洽错误比例。

6. **distinguish 把中位数错误地当成可加运算。**  
   **类型：**数学错误、隐含假设。**严重度：严重。**  
   **证据：**[distinguish.py:60](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/distinguish.py:60)、[distinguish.py:64](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/distinguish.py:64) 用注入向量的中位数计算比较序列的偏移；实际序列在 [distinguish.py:81](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/distinguish.py:81) 对“原始值＋注入”重新取中位数。反例：原节点 `[0,1,100]`，注入 `[10,0,0]`，共识实际偏移 **9**，`shift_of` 给 **0**。因此其 μ 并不普遍等于实际统计量的注入偏移。  
   **判决检验：**对每个场景比较 `shift_of` 与无噪声及含噪声比较序列的实际位移，单独报告由此造成的预测误差。

7. **inject 的污染进入了下一批第①步，却被评分为未污染。**  
   **类型：**假失败、代码缺陷。**严重度：严重。**  
   **证据：**[team2.py:97](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/team2.py:97) 保存讨论后的留言，[team2.py:92](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/team2.py:92) 将它送入下一批独立估计；[score.py:32](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/score.py:32) 仍将 inject 的全部 agent 视为未污染，并在第 62 行计误报。实际缓存 [b4_s1_ds_pro_b.json:1](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/runs_A/inject_1000/b4_s1_ds_pro_b.json:1) 已写明 **“Applied +40% distance”**。这直接违反“inject 第①步不受影响”的设计。  
   **判决检验：**逐批检查 inject 的第①步提示和输出，并将来自历史留言的污染与真正误报分开评分。

8. **data3 没有测试“污染过半”。**  
   **类型：**隐含假设、基线不公平。**严重度：中等。**  
   **证据：**[team2.py:26](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/team2.py:26) 只有 6 个独立报告 agent，第 29 行污染其中 3 个；[score.py:14](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/score.py:14) 却称其为“污染过半”。实际是 **3/6，恰好一半**。  
   **判决检验：**按独立报告者人数核算污染比例，另测至少 4/6 后再判断多数污染下的基线表现。

9. **校准期曾报警的 agent，后续报警会被永久抹掉。**  
   **类型：**代码缺陷、假通过／假失败。**严重度：中等。**  
   **证据：**[commonmode.py:64](/Users/sly/Desktop/agent/claude/ring-arith-toy/switching/commonmode.py:64) 只保留第一次报警；[referee.py:67](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/referee.py:67) 将校准期首次报警映射成 `None`，后续评估期报警无法再出现。同伴基线的处理方式不同。合成记录已复现：原始首次报警在索引 **28**，评估期持续偏移，最终却报告该 agent **None**。未据此断言现有正式局已受影响。  
   **判决检验：**构造一次校准期越线、随后评估期持续越线的序列，要求返回评估期报警并与同伴基线采用相同规则。

10. **SLQA 实验室筛选使用了整个评估期的可用性。**  
    **类型：**数据泄漏、隐含假设。**严重度：中等。**  
    **证据：**[ladder2.py:71](/Users/sly/Desktop/agent/claude/ring-arith-toy/theory/ladder2.py:71) 要求实验室在全部 `items` 中覆盖率 ≥60%，包括测试段。只用沙盒条件有 **21** 个候选，加入未来覆盖率后剩 **17** 个，排除了 CAGM、CN、MBNEP、MDPH。因此结果针对长期留存的筛选子集。  
    **判决检验：**只按沙盒冻结候选实验室，保留测试期缺失和退出情况，检查判定是否改变。

11. **簇抽样只进入预测区间，没有进入诚实概率的标准误。**  
    **类型：**隐含假设、假通过／假失败。**严重度：中等。**  
    **证据：**[law_v6.py:97](/Users/sly/Desktop/agent/claude/ring-arith-toy/switching/law_v6.py:97) 始终除以逐题 `len(bad)`；[validate_v61.py:82](/Users/sly/Desktop/agent/claude/ring-arith-toy/validate_v61.py:82) 的轮次分块只交给外围区间。重复同一批 20 条观测 50 次、没有增加信息的反例中，某节点诚实概率从 **0.8265 降到 0.00000576**。FL 同轮 50 个坐标共享模型和服务器资料，不能直接视为 50 次独立身份测量。  
    **判决检验：**固定观测信息，仅复制簇内数据，检查身份后验是否保持稳定，并用轮级标准误重算识别结果。

12. **当前“预测”使用整个评估窗拟合自身，再评分同一窗。**  
    **类型：**隐含假设、评估期信息复用。**严重度：中等。**  
    **证据：**[law_v6.py:74](/Users/sly/Desktop/agent/claude/ring-arith-toy/switching/law_v6.py:74) 使用全部评估期估节点噪声，第 95 行开始用全部评估期估身份；[law_v6.py:115](/Users/sly/Desktop/agent/claude/ring-arith-toy/switching/law_v6.py:115) 随即给这些题计算概率。两个验证器及 [referee.py:77](/Users/sly/Desktop/agent/claude/ring-arith-toy/mas/referee.py:77) 都如此。它支持的是**全窗回顾风险估计**，尚未验证逐题在线预测；“零拟合”需要明确限于公式未做全局拟合。  
    **判决检验：**使用严格前缀或独立历史窗估计身份与噪声，预测后续未参与估计的题，再比较 MAE、ECE 和识别率。

经得起这轮攻击的部分：

- 从保存结果复算，Fashion-MNIST MAE **0.070928**、BIPM **0.071468**；正确处理并列秩后，Spearman 仍为 **0.94045、0.85370**，主要数字没有被排序实现推翻。
- 事件提升度 **2.36、5.70** 可以复现；保留各实验室事件基率的组内置换后仍显著，p 约 **0.0015、0.0005**。
- FL、BIPM 和 MAS 中共同平移真值的主要比较路径可以抵消真值；SLQA 的逐瓶归一化属于不同情况。

预注册方面，提交链支持“先登记、后记录结果”：

| 实验 | 预注册 → 结果记录 |
|---|---|
| Fashion v6.1 | `74ce4dc` → `11dbe68` |
| BIPM | `6179a7e` → `2714a35` |
| 事件 | `08cdca4` → `eafdc2c` |
| 约束 | `eafdc2c` → `8936a21` |
| 阶梯 | `fd650a3` → `b46fc90` |
| distinguish 第二轮 | `79374ca` → `1245c35` |

这些对应提交间的算法修改未见偷换判据；冻结标签到当前的相关算法也没有差异。**Git 提交顺序无法独立证明实际首次运行时间，目前没有足够证据指控先跑后提交。**

最后，最可能错的三件事：

1. **“本实验显示单秤不能归因”**：保留符号即可在现有实验中达到 100%。
2. **“相对误差 >20% 时约束 AUC 0.72，通过门槛”**：按该文字定义重算为 0.690。
3. **“MAD 三角帽能可靠识别共有误差”**：完全独立的来源已能触发强烈假共有误差和拒答。