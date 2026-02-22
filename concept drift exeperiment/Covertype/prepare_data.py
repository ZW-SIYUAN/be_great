"""
prepare_data.py
───────────────
下载 Forest Covertype 数据集，将 4 个 one-hot 的 Wilderness_Area 列合并为单一
类别列（1-4），将 40 个 one-hot 的 Soil_Type 列合并为单一类别列（1-40），
生成适合 GReaT 训练的 CSV 文件。

输出文件：
  covertype.csv        全量数据（581 012 行，13 列）
  covertype_train.csv  均衡训练集（每个区域各取 2000 行 = 共 8000 行）

运行：
  cd "concept drift exeperiment/Covertype"
  python prepare_data.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype

HERE = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(HERE, filename)

# ── 1. 下载 ──────────────────────────────────────────────────────
print("正在下载 Covertype 数据集（首次运行需联网）...")
bunch = fetch_covtype(as_frame=True)
df = bunch.frame.copy()
print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

# ── 2. 整理列名 ───────────────────────────────────────────────────
wa_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
st_cols = [c for c in df.columns if c.startswith("Soil_Type")]

# one-hot → 单一类别列（argmax 返回 0-based，+1 转为 1-based）
df["Wilderness_Area"] = (df[wa_cols].values.argmax(axis=1) + 1).astype(str)
df["Soil_Type"]       = (df[st_cols].values.argmax(axis=1) + 1).astype(str)
df["Cover_Type"]      = df["Cover_Type"].astype(str)

df = df.drop(columns=wa_cols + st_cols)

NUM_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
df = df[NUM_COLS + ["Wilderness_Area", "Soil_Type", "Cover_Type"]]

# ── 3. 数据概览 ───────────────────────────────────────────────────
print(f"\n整理后: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"\n列名: {list(df.columns)}")
print(f"\nWilderness_Area 分布（空间漂移维度）:")
print(df["Wilderness_Area"].value_counts().sort_index().to_string())
print(f"\nCover_Type 分布（预测目标）:")
print(df["Cover_Type"].value_counts().sort_index().to_string())
print(f"\n数值特征统计:")
print(df[NUM_COLS].describe().round(2).to_string())

# ── 4. 保存全量数据 ───────────────────────────────────────────────
df.to_csv(data_path("covertype.csv"), index=False)
print(f"\n[OK] 已保存 covertype.csv  ({len(df)} 行)")

# ── 5. 构建均衡训练集（每区域 2000 行）────────────────────────────
N_PER_AREA = 2000
parts = []
for area in sorted(df["Wilderness_Area"].unique(), key=int):
    area_df = df[df["Wilderness_Area"] == area]
    n = min(len(area_df), N_PER_AREA)
    parts.append(area_df.sample(n=n, random_state=42))
    print(f"  Area {area}: 原始 {len(area_df):6d} 行  →  采样 {n} 行")

train_df = (pd.concat(parts)
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))

train_df.to_csv(data_path("covertype_train.csv"), index=False)
print(f"\n[OK] 已保存 covertype_train.csv  ({len(train_df)} 行)")
print(f"     各区域: { {k: int(v) for k, v in train_df['Wilderness_Area'].value_counts().sort_index().items()} }")
print(f"     各类别: { {k: int(v) for k, v in train_df['Cover_Type'].value_counts().sort_index().items()} }")
