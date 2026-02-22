import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# 加载数据（各取前2000行，与 plot.py 保持一致）
real = pd.read_csv("electricity.csv").iloc[:2000]
synth = pd.read_csv("electricity_synthetic.csv").iloc[:2000]

cat_cols = ["class"]
num_cols = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]

# ---------- 1. 统计指标 ----------
print("=== ColumnShapes ===")
print(ColumnShapes().compute(real, synth, num_cols=num_cols, cat_cols=cat_cols))

print("\n=== ColumnPairTrends ===")
print(ColumnPairTrends().compute(real, synth, num_cols=num_cols, cat_cols=cat_cols))

print("\n=== BasicStatistics ===")
result = BasicStatistics().compute(real, synth, num_cols=num_cols, cat_cols=cat_cols)
for col, stats in result["basic_statistics"].items():
    print(f"  {col}: {stats}")

# ---------- 2. 效用指标 ----------
print("\n=== MLEfficiency (train on synth, test on real) ===")
mle = MLEfficiency(
    model=RandomForestClassifier,
    metric=accuracy_score,
    model_params={"n_estimators": 100},
).compute(real, synth, label_col="class", cat_cols=cat_cols, num_cols=num_cols)
print(mle)

print("\n=== DiscriminatorMetric (closer to 0.5 is better) ===")
disc = DiscriminatorMetric(n_runs=5).compute(real, synth, cat_cols=cat_cols)
print(disc)

# ---------- 3. 隐私指标 ----------
print("\n=== DistanceToClosestRecord ===")
dcr = DistanceToClosestRecord().compute(real, synth, num_cols=num_cols, cat_cols=cat_cols)
print({k: v for k, v in dcr.items() if k != "distances"})  # 省略完整距离列表

print("\n=== kAnonymization ===")
print(kAnonymization().compute(real, synth))

print("\n=== lDiversity (sensitive_col='class') ===")
print(lDiversity(sensitive_col="class").compute(real, synth))

print("\n=== IdentifiabilityScore ===")
print(IdentifiabilityScore().compute(real, synth, num_cols=num_cols, cat_cols=cat_cols))

print("\n=== DeltaPresence ===")
print(DeltaPresence(threshold=0.0).compute(real, synth, num_cols=num_cols, cat_cols=cat_cols))

print("\n=== MembershipInference (closer to 0.5 is better) ===")
print(MembershipInference().compute(real, synth, num_cols=num_cols, cat_cols=cat_cols))