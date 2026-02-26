import pandas as pd
from be_great import GReaT

# ── 1. 加载数据 ──────────────────────────────────────────────
df = pd.read_csv("electricity.csv")
print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(df.dtypes)

# ── 2. 初始化模型 ─────────────────────────────────────────────
model = GReaT(
    llm="tabularisai/Qwen3-0.3B-distil",
    epochs=40,
    batch_size=32,
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

# ── 3. 微调 ───────────────────────────────────────────────────
model.fit(df)

# ── 4. 保存模型 ───────────────────────────────────────────────
model.save("electricity_great_model")
print("模型已保存至 electricity_great_model/")

# ── 5. 生成 2000 个合成样本 ───────────────────────────────────
synthetic_df = model.sample(
    n_samples=2000,
    guided_sampling=True,
    random_feature_order=True,
    max_length=200,
    temperature=0.7,
)

print(f"\n合成数据: {synthetic_df.shape[0]} 行")
print(synthetic_df.head())
print("\nclass 分布:")
print(synthetic_df["class"].value_counts())

# ── 6. 保存合成数据 ───────────────────────────────────────────
synthetic_df.to_csv("electricity_synthetic.csv", index=False)
print("\n合成数据已保存至 electricity_synthetic.csv")
