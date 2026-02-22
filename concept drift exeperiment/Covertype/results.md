Real : (8000, 13),  Synth: (8000, 13)
Real Cover_Type:
Cover_Type
1    2749
2    2893
3    1276
4     143
5      71
6     579
7     289
Name: count, dtype: int64

Synth Cover_Type:
Cover_Type
1     2531
2     3890
3      911
4       38
5       14
6      471
7      137
8        2
9        1
10       1
14       1
19       2
22       1
Name: count, dtype: int64

============================================================
                          1. 统计指标
============================================================

── ColumnShapes（各列分布相似度，↑越接近1越好）──
  均值: 0.8835  标准差: 0.0394
  各列得分:
    Aspect                                         0.8824
    Cover_Type                                     0.8744
    Elevation                                      0.8889
    Hillshade_3pm                                  0.9078
    Hillshade_9am                                  0.9453
    Hillshade_Noon                                 0.9201
    Horizontal_Distance_To_Fire_Points             0.8313
    Horizontal_Distance_To_Hydrology               0.9576
    Horizontal_Distance_To_Roadways                0.8236
    Slope                                          0.8718
    Soil_Type                                      0.8390
    Vertical_Distance_To_Hydrology                 0.8774
    Wilderness_Area                                0.8656

── ColumnPairTrends（列间相关性保留度，↑越接近1越好）──
  总体均值:   0.8201
  数值对均值: 0.8250
  类别对均值: 0.7457

── BasicStatistics（各列基本统计量对比）──
  Elevation                                      real_mean=  2880.286  synth_mean=  2935.066  diff%=  1.90
  Aspect                                         real_mean=   168.671  synth_mean=   180.834  diff%=  7.21
  Slope                                          real_mean=    15.443  synth_mean=    16.231  diff%=  5.11
  Horizontal_Distance_To_Hydrology               real_mean=   261.891  synth_mean=   266.438  diff%=  1.74
  Vertical_Distance_To_Hydrology                 real_mean=    48.110  synth_mean=    61.271  diff%= 27.36
  Horizontal_Distance_To_Roadways                real_mean=  1713.960  synth_mean=  1948.865  diff%= 13.71
  Hillshade_9am                                  real_mean=   207.577  synth_mean=   209.265  diff%=  0.81
  Hillshade_Noon                                 real_mean=   221.091  synth_mean=   222.251  diff%=  0.52
  Hillshade_3pm                                  real_mean=   144.533  synth_mean=   146.322  diff%=  1.24
  Horizontal_Distance_To_Fire_Points             real_mean=  1747.502  synth_mean=  1926.528  diff%= 10.24
  Wilderness_Area                                real_top=2(25.00%)  synth_top=1(38.42%)
  Soil_Type                                      real_top=10(13.58%)  synth_top=29(18.89%)
  Cover_Type                                     real_top=2(36.16%)  synth_top=2(48.62%)

============================================================
                          2. 效用指标
============================================================

── MLEfficiency（在合成数据上训练，在真实数据上测试，↑越高越好）──
  mle_mean: 0.5115  mle_std: 0.0039
  各次得分: ['0.5162', '0.5112', '0.5075', '0.5069', '0.5156']

── DiscriminatorMetric（↓越接近0.5越难区分，合成质量越好）──
  discriminator_mean: 0.9640  discriminator_std: 0.0015

============================================================
                          3. 隐私指标
============================================================

── DistanceToClosestRecord（DCR，↑均值越大，n_copies=0 最佳）──
  dcr_mean:    544.2178
  dcr_std:     577.4102
  dcr_min:     111.0000
  n_copies:    0  (0.00%)

── kAnonymization（k_ratio≥1 说明合成数据匿名性不低于真实）──
  k_real:      122
  k_synthetic: 1
  k_ratio:     0.0082

── lDiversity（sensitive_col='Cover_Type'，l_ratio≥1 最佳）──
  l_real:      2
  l_synthetic: 1
  l_ratio:     0.5000

── IdentifiabilityScore（↓越低隐私越好）──
  identifiability_score: 0.0076
  mean_distance_ratio:   0.8922

── DeltaPresence（↓越低越好，存在推断风险）──
  delta_presence:       0.0000
  mean_nearest_distance:198.5763

── MembershipInference（↓越接近0.5越好，0.5=攻击无效）──
  membership_inference_score:  0.5095
  mean_member_distance:        197.5872
  mean_non_member_distance:    199.5652

1. 致命缺陷：标签“幻觉”（Hallucination）
这是报告中最醒目的问题。请看 Cover_Type 的分布：
真实数据： 只有 1 到 7 类。
合成数据： 出现了 8, 9, 10, 14, 19, 22 等原数据中根本不存在的类别！
分析：
LLM（如 BE GREAT）是将数值视为 Token 处理的。由于它在预训练阶段见过大量的数字序列，当它在生成时，如果概率分布稍微偏移，它就会根据“直觉”生成看起来像标签的数字。
后果： 这直接导致了 DiscriminatorMetric 高达 0.964。判别器只要看到类别 > 7，就能 100% 判定这是假数据。这也是为什么你的 MLEfficiency 只有 0.51 的核心原因——分类器在合成数据上学到了错误的类别。

2. 统计指标：表面的繁荣
ColumnShapes (0.8835): 看起来不错，说明单列的概率密度曲线与原图比较贴合。
ColumnPairTrends (0.8201): 表现尚可，说明模型捕捉到了变量间的基本物理关系（例如海拔 Elevation 与植被类型 Cover_Type 的关联）。
BasicStatistics 的警示：
Vertical_Distance_To_Hydrology 偏差高达 27.36%。
Soil_Type 的 Top 1 类别从 10 变成了 29。
这说明模型在处理长尾分布（即那些样本量很少的特征值）时表现很差。

3. 效用指标：不及格的实战表现
MLEfficiency (0.5115): * 这是一个非常糟糕的分数。对于 Covertype 数据集，即使是简单的随机森林模型，在真实数据上的准确率通常也能达到 0.75 - 0.85。
0.51 意味着合成数据丢失了将近 30%-40% 的预测信息量。它学到了“形”，但没学到驱动分类的核心“神”。



=================================================================
                       Quantitative Summary
=================================================================

[A] Within-Area KS Test (Real_area vs Synth_area) - Lower is more accurate
  Feature                                   Area1  Area2  Area3  Area4
  Elevation                                 0.196  0.174  0.246  0.183
  Aspect                                    0.220  0.077  0.124  0.163
  Slope                                     0.253  0.217  0.103  0.302
  Horizontal_Distance_To_Hydrology          0.025  0.161  0.066  0.121
  Vertical_Distance_To_Hydrology            0.174  0.185  0.121  0.085
  Horizontal_Distance_To_Roadways           0.349  0.375  0.138  0.432
  Hillshade_9am                             0.165  0.072  0.033  0.284
  Hillshade_Noon                            0.086  0.155  0.162  0.260
  Hillshade_3pm                             0.110  0.054  0.062  0.190
  Horizontal_Distance_To_Fire_Points        0.220  0.109  0.104  0.504

[B] Correlation of Inter-area KS Matrices (Real Matrix vs Synth Matrix)
    r ≈ 1 means GReaT preserved the relative spatial structure
  Elevation                                 r=+0.9564  p=2.82e-03
  Slope                                     r=+0.8200  p=4.57e-02
  Horizontal_Distance_To_Hydrology          r=-0.1747  p=7.41e-01
  Hillshade_9am                             r=+0.3469  p=5.00e-01
  Horizontal_Distance_To_Fire_Points        r=+0.9394  p=5.39e-03

[C] Cross-Area Classification Accuracy Summary
  Real  Diagonal (In-area) Mean: 0.7994
  Synth Diagonal (In-area) Mean: 0.4680
  Real  Off-diagonal (Cross-area) Mean: 0.3514 (Gap: 0.4480)
  Synth Off-diagonal (Cross-area) Mean: 0.3368 (Gap: 0.1313)

[D] Global Cover_Type TVD (Total Variation Distance, lower is better)
  Area 1 (Rawah         )  TVD = 0.1077
  Area 2 (Neota         )  TVD = 0.2298
  Area 3 (Comanche
Peak )  TVD = 0.0892
  Area 4 (Cache la
Poudre)  TVD = 0.2745


优势：宏观结构的“地理学家”
在两个实验中，GReaT 都表现出了极其惊人的结构保留能力。
在 Covertype 中，区域 1 到区域 4 的海拔（Elevation）高低次序、坡度变化，GReaT 几乎完美复刻（相关性 r>0.9）。
这说明 LLM 在处理“A 区域通常比 B 区域高”这种关系型知识时，比传统的 GAN 更有优势。它理解的是数据背后的“格局”。
劣势：微观判别的“近视眼”
虽然宏观轮廓对，但到了预测任务（MLEfficiency）时，模型却表现不佳。

精度瓶颈： 电力数据中的 date 偏移和 Covertype 中的 0.47 准确率，反映了 LLM 在 Token 化数值时，丢失了细微的特征边界。
**过平滑（Over-smoothing）**： 合成数据往往比真实数据更“中庸”。在 Covertype 中，它把四个区域的特征合成得太像了（Gap 缩小），导致它丢掉了那些能区分“冷杉”和“云杉”的决定性细微差异。

3. 症结分析：为什么会出现“幻觉类别”？
在 Covertype 中看到的类别 8, 9, 10... 是 LLM 合成表格数据的典型副作用：
自回归的随机性： LLM 是按概率预测下一个 Token 的。如果训练不够充分，或者概率分布较平，它会根据预训练时见到的数字序列“随口”说出一个数字。
缺乏约束： 传统的合成模型（如 CTGAN）会固定类别范围，但 LLM 认为“数字就是文本”，它并不理解在这个语境下是非法的。