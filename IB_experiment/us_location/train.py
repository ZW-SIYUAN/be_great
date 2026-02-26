import pandas as pd
from pathlib import Path
from be_great import GReaT

HERE     = Path(__file__).parent
MODEL_ID = "gpt2-medium"

# ── 1. 加载数据 ──────────────────────────────────────────────
df = pd.read_csv(HERE / "us_location_train.csv")
print(f"训练数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(df.dtypes)
print(f"\nstate_code 分布 (前10):\n{df['state_code'].value_counts().head(10)}")

# ── 2. 初始化模型 ─────────────────────────────────────────────
# 数据集特征：16320行 / 5列 / 每行文本极短(≈40 token)
# batch_size=16：列少文本短，显存充裕
# max_length=128：5列文本远短于 sick 的30列，128 token 足够
# float_precision=2：lat/lon 保留2位小数与原始数据一致
model = GReaT(
    llm=MODEL_ID,
    epochs=15,
    batch_size=64,
    float_precision=2,
    bf16=True,
    dataloader_num_workers=0,
    efficient_finetuning="lora",
    lora_config={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["c_attn", "c_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
)

# ── 3. 微调 ───────────────────────────────────────────────────
model.fit(df)

# ── 4. 保存模型 ──────────────────────────────────────────────
model_dir = HERE / "us_great_model"
model.save(str(model_dir))
print(f"模型已保存至 {model_dir}")

# ── 5. 生成合成样本（与训练集等量）────────────────────────────
synthetic_df = model.sample(
    n_samples=16320,
    guided_sampling=True,
    random_feature_order=True,
    max_length=128,
    temperature=0.7,
)

print(f"\n合成数据: {synthetic_df.shape[0]} 行, {synthetic_df.shape[1]} 列")
print(synthetic_df.head())
print(f"\nstate_code 分布 (前10):\n{synthetic_df['state_code'].value_counts().head(10)}")

# ── 6. 保存合成数据 ──────────────────────────────────────────
out_path = HERE / "us_location_synthetic.csv"
synthetic_df.to_csv(out_path, index=False)
print(f"\n合成数据已保存至 {out_path}")
