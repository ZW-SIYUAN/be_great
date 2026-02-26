"""
evaluate_quality.py
───────────────────
使用库内置指标函数对 covertype_synthetic.csv 进行全面质量评估，
对标 electricity 实验中的 evaluate.py，结构完全一致。

前置条件：先运行 train_and_sample.py 生成 covertype_synthetic.csv

运行：
  cd "concept drift exeperiment/Covertype"
  python evaluate_quality.py
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(HERE, filename)
from sklearn.metrics import accuracy_score

from be_great.metrics import (
    ColumnShapes,
    ColumnPairTrends,
    BasicStatistics,
    MLEfficiency,
    DiscriminatorMetric,
    DistanceToClosestRecord,
    kAnonymization,
    lDiversity,
    IdentifiabilityScore,
    DeltaPresence,
    MembershipInference,
)

# ── 数据加载 ──────────────────────────────────────────────────────
# 与 electricity 保持一致：各取前 8000 行作为评估集
real  = pd.read_csv(data_path("covertype_train.csv")).iloc[:8000].reset_index(drop=True)
synth = pd.read_csv(data_path("covertype_synthetic.csv")).iloc[:8000].reset_index(drop=True)

print(f"Real : {real.shape},  Synth: {synth.shape}")
print(f"Real Cover_Type:\n{real['Cover_Type'].value_counts().sort_index()}")
print(f"\nSynth Cover_Type:\n{synth['Cover_Type'].value_counts().sort_index()}")

NUM_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
CAT_COLS = ["Wilderness_Area", "Soil_Type", "Cover_Type"]

# 确保数值列为 float（LLM 可能输出 object 类型）
for col in NUM_COLS:
    real[col]  = pd.to_numeric(real[col],  errors="coerce")
    synth[col] = pd.to_numeric(synth[col], errors="coerce")

SEP = "=" * 60

# ════════════════════════════════════════════════════════════════
# 1. 统计指标
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n{'1. 统计指标':^60}\n{SEP}")

print("\n── ColumnShapes（各列分布相似度，↑越接近1越好）──")
shapes = ColumnShapes().compute(real, synth, num_cols=NUM_COLS, cat_cols=CAT_COLS)
print(f"  均值: {shapes['column_shapes_mean']:.4f}  标准差: {shapes['column_shapes_std']:.4f}")
print("  各列得分:")
for col, score in sorted(shapes["column_shapes_detail"].items()):
    print(f"    {col:45s}  {score:.4f}")

print("\n── ColumnPairTrends（列间相关性保留度，↑越接近1越好）──")
trends = ColumnPairTrends().compute(real, synth, num_cols=NUM_COLS, cat_cols=CAT_COLS)
print(f"  总体均值:   {trends['column_pair_trends_mean']:.4f}")
print(f"  数值对均值: {trends['column_pair_trends_numerical']:.4f}")
print(f"  类别对均值: {trends['column_pair_trends_categorical']:.4f}")

print("\n── BasicStatistics（各列基本统计量对比）──")
stats_result = BasicStatistics().compute(real, synth, num_cols=NUM_COLS, cat_cols=CAT_COLS)
for col, s in stats_result["basic_statistics"].items():
    if "real_mean" in s:
        print(f"  {col:45s}  real_mean={s['real_mean']:10.3f}  synth_mean={s['synth_mean']:10.3f}"
              f"  diff%={s['mean_diff_pct']:6.2f}")
    else:
        real_top = max(s["real_distribution"], key=s["real_distribution"].get)
        synth_top = max(s["synth_distribution"], key=s["synth_distribution"].get)
        print(f"  {col:45s}  real_top={real_top}({s['real_distribution'][real_top]:.2%})"
              f"  synth_top={synth_top}({s['synth_distribution'].get(synth_top, 0):.2%})")

# ════════════════════════════════════════════════════════════════
# 2. 效用指标
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n{'2. 效用指标':^60}\n{SEP}")

print("\n── MLEfficiency（在合成数据上训练，在真实数据上测试，↑越高越好）──")
mle = MLEfficiency(
    model=RandomForestClassifier,
    metric=accuracy_score,
    model_params={"n_estimators": 100},
).compute(real, synth, label_col="Cover_Type", cat_cols=CAT_COLS, num_cols=NUM_COLS)
print(f"  mle_mean: {mle['mle_mean']:.4f}  mle_std: {mle['mle_std']:.4f}")
print(f"  各次得分: {[f'{s:.4f}' for s in mle['mle_scores']]}")

print("\n── DiscriminatorMetric（↓越接近0.5越难区分，合成质量越好）──")
disc = DiscriminatorMetric(n_runs=5).compute(real, synth, cat_cols=CAT_COLS)
print(f"  discriminator_mean: {disc['discriminator_mean']:.4f}"
      f"  discriminator_std: {disc['discriminator_std']:.4f}")

# ════════════════════════════════════════════════════════════════
# 3. 隐私指标
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n{'3. 隐私指标':^60}\n{SEP}")

print("\n── DistanceToClosestRecord（DCR，↑均值越大，n_copies=0 最佳）──")
dcr = DistanceToClosestRecord().compute(real, synth, num_cols=NUM_COLS, cat_cols=CAT_COLS)
print(f"  dcr_mean:    {dcr['dcr_mean']:.4f}")
print(f"  dcr_std:     {dcr['dcr_std']:.4f}")
print(f"  dcr_min:     {dcr['dcr_min']:.4f}")
print(f"  n_copies:    {dcr['n_copies']}  ({dcr['ratio_copies']:.2%})")

print("\n── kAnonymization（k_ratio≥1 说明合成数据匿名性不低于真实）──")
kanon = kAnonymization().compute(real, synth)
print(f"  k_real:      {kanon['k_real']}")
print(f"  k_synthetic: {kanon['k_synthetic']}")
print(f"  k_ratio:     {kanon['k_ratio']:.4f}")

print("\n── lDiversity（sensitive_col='Cover_Type'，l_ratio≥1 最佳）──")
ldiv = lDiversity(sensitive_col="Cover_Type").compute(real, synth)
print(f"  l_real:      {ldiv['l_real']}")
print(f"  l_synthetic: {ldiv['l_synthetic']}")
print(f"  l_ratio:     {ldiv['l_ratio']:.4f}")

print("\n── IdentifiabilityScore（↓越低隐私越好）──")
ident = IdentifiabilityScore().compute(real, synth, num_cols=NUM_COLS, cat_cols=CAT_COLS)
print(f"  identifiability_score: {ident['identifiability_score']:.4f}")
print(f"  mean_distance_ratio:   {ident['mean_distance_ratio']:.4f}")

print("\n── DeltaPresence（↓越低越好，存在推断风险）──")
delta = DeltaPresence(threshold=0.0).compute(real, synth, num_cols=NUM_COLS, cat_cols=CAT_COLS)
print(f"  delta_presence:       {delta['delta_presence']:.4f}")
print(f"  mean_nearest_distance:{delta['mean_nearest_distance']:.4f}")

print("\n── MembershipInference（↓越接近0.5越好，0.5=攻击无效）──")
mi = MembershipInference().compute(real, synth, num_cols=NUM_COLS, cat_cols=CAT_COLS)
print(f"  membership_inference_score:  {mi['membership_inference_score']:.4f}")
print(f"  mean_member_distance:        {mi['mean_member_distance']:.4f}")
print(f"  mean_non_member_distance:    {mi['mean_non_member_distance']:.4f}")

print(f"\n{SEP}")
print("质量评估完成。")
