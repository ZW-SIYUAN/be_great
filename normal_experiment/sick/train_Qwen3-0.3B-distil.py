import pandas as pd
from pathlib import Path
from be_great import GReaT

HERE = Path(__file__).parent

# ── 1. 加载数据 ──────────────────────────────────────────────
df = pd.read_csv(HERE / "data" / "sick_train.csv")
print(f"训练数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(df.dtypes)
print(f"\nClass 分布:\n{df['Class'].value_counts()}")
print(f"正例率 (sick): {df['Class'].eq('sick').mean()*100:.2f}%")

# ── 2. 初始化模型 ─────────────────────────────────────────────
# 数据集特征：3016行 / 30列 / 23类别列 / 严重不平衡 (sick≈6%)
# 序列长度估算：30列 × ~10 token/列 ≈ 300 token，设 max_length=512
# RTX5080 16GB：0.3B模型 bf16 <1GB，batch32 × seq512 激活约3GB，安全
model = GReaT(
    llm="tabularisai/Qwen3-0.3B-distil",
    epochs=50,           # 3016/32≈94步/epoch，100epoch≈9400步，充足
    batch_size=8,        # 30列文本较长(≈300-400 token)，从64降至32保证稳定
    float_precision=3,    # TSH/T3/TT4等连续值保留3位小数
    bf16=True,
    dataloader_num_workers=0,
    efficient_finetuning="lora",
    lora_config={
        "r": 16,          # 30列复杂结构，r=16保持充足表达能力
        "lora_alpha": 32, # alpha/r = 2，标准比例
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
)

# ── 3. 微调 ───────────────────────────────────────────────────
model.fit(df)

# ── 4. 保存模型 ───────────────────────────────────────────────
model.save(str(HERE / "sick_great_model"))
print("模型已保存至 sick_great_model/")

# ── 5. 生成 3016 个合成样本（与训练集等量，符合论文设置）────────
synthetic_df = model.sample(
    n_samples=3016,
    guided_sampling=True,
    random_feature_order=True,
    max_length=512,       # 30列文本长，512 token 保证完整生成
    temperature=0.7,
)

print(f"\n合成数据: {synthetic_df.shape[0]} 行, {synthetic_df.shape[1]} 列")
print(synthetic_df.head())
print("\nClass 分布:")
print(synthetic_df["Class"].value_counts())
if "Class" in synthetic_df.columns:
    print(f"正例率 (sick): {synthetic_df['Class'].eq('sick').mean()*100:.2f}%")

# ── 6. 保存合成数据 ───────────────────────────────────────────
out_path = HERE / "data" / "sick_synthetic.csv"
synthetic_df.to_csv(out_path, index=False)
print(f"\n合成数据已保存至 {out_path}")
