"""
train_and_sample.py
───────────────────
使用 GReaT 对 Covertype 训练集进行微调，然后生成 2000 条合成数据。

前置条件：先运行 prepare_data.py 生成 covertype_train.csv

输出文件：
  covertype_great_model/   训练后的模型
  covertype_synthetic.csv  合成数据（≈2000 行）

运行：
  cd "concept drift exeperiment/Covertype"
  python train_and_sample.py
"""

import os
import pandas as pd
from be_great import GReaT

HERE = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(HERE, filename)

# ── 1. 加载数据 ───────────────────────────────────────────────────
df = pd.read_csv(data_path("covertype_train.csv"))
print(f"训练数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(df.dtypes)
print(f"\nWilderness_Area 分布:\n{df['Wilderness_Area'].value_counts().sort_index()}")
print(f"\nCover_Type 分布:\n{df['Cover_Type'].value_counts().sort_index()}")

# ── 2. 初始化模型 ─────────────────────────────────────────────────
model = GReaT(
    llm="tabularisai/Qwen3-0.3B-distil",
    epochs=40,
    batch_size=8,
    float_precision=3,
    bf16=True,
    dataloader_num_workers=0,
    efficient_finetuning="lora",
    lora_config={
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
)

# ── 3. 微调 ───────────────────────────────────────────────────────
model.fit(df)

# ── 4. 保存模型 ───────────────────────────────────────────────────
model.save(data_path("covertype_great_model"))
print("\n模型已保存至 covertype_great_model/")

# ── 5. 生成 8000 条合成样本 ───────────────────────────────────────
# max_length=300：Covertype 每行约 12 列，文本较长，适当放宽
synthetic_df = model.sample(
    n_samples=8000,
    guided_sampling=True,
    random_feature_order=True,
    max_length=300,
    temperature=0.7,
)

print(f"\n合成数据: {synthetic_df.shape[0]} 行, {synthetic_df.shape[1]} 列")
print(synthetic_df.head())

print(f"\nCover_Type 分布:\n{synthetic_df['Cover_Type'].value_counts().sort_index()}")
print(f"\nWilderness_Area 分布:\n{synthetic_df['Wilderness_Area'].value_counts().sort_index()}")

# ── 6. 保存合成数据 ───────────────────────────────────────────────
synthetic_df.to_csv(data_path("covertype_synthetic.csv"), index=False)
print("\n合成数据已保存至 covertype_synthetic.csv")
