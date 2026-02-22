import pandas as pd
from pathlib import Path
from be_great import GReaT

HERE = Path(__file__).parent

# ── 1. 加载数据 ──────────────────────────────────────────────
df = pd.read_csv(HERE / "travel_train.csv")
print(f"训练数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(df.dtypes)
print(f"\nTarget 分布:\n{df['Target'].value_counts()}")

# ── 2. 初始化模型 ─────────────────────────────────────────────
model = GReaT(
    llm="tabularisai/Qwen3-0.3B-distil",
    epochs=150,          # 763行 / batch64 ≈ 12步/epoch，150epoch≈1800步，足够收敛
    batch_size=64,       # RTX5080 16GB显存，0.3B模型占用<1GB，64完全没有压力
    float_precision=3,
    bf16=True,           # Blackwell架构原生支持bf16，速度快且精度足
    dataloader_num_workers=0,  # Windows下保持0，避免多进程问题
    efficient_finetuning="lora",
    lora_config={
        "r": 16,             # 从8提升到16，表达能力更强，显存开销可忽略
        "lora_alpha": 32,    # 保持 alpha/r = 2 的标准比例
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
)

# ── 3. 微调 ───────────────────────────────────────────────────
model.fit(df)

# ── 4. 保存模型 ───────────────────────────────────────────────
model.save(str(HERE / "travel_great_model"))
print("模型已保存至 travel_great_model/")

# ── 5. 生成 763 个合成样本（与训练集等量，符合论文设置）─────────
synthetic_df = model.sample(
    n_samples=763,
    guided_sampling=True,
    random_feature_order=True,
    max_length=200,      # travel数据集7列，文本较短，200足够
    temperature=0.7,
)

print(f"\n合成数据: {synthetic_df.shape[0]} 行")
print(synthetic_df.head())
print("\nTarget 分布:")
print(synthetic_df["Target"].value_counts())

# ── 6. 保存合成数据 ───────────────────────────────────────────
synthetic_df.to_csv(HERE / "travel_synthetic.csv", index=False)
print(f"\n合成数据已保存至 {HERE / 'travel_synthetic.csv'}")
