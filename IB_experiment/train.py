import pandas as pd
from pathlib import Path
from be_great import GReaT

HERE     = Path(__file__).parent
MODEL_ID = "gpt2-medium"

# ── 1. 加载数据 ──────────────────────────────────────────────
df = pd.read_csv(HERE / "data" / "sick_train.csv")
print(f"训练数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(df.dtypes)
print(f"\nClass 分布:\n{df['Class'].value_counts()}")
print(f"正例率 (sick): {df['Class'].eq('sick').mean()*100:.2f}%")

# ── 2. 初始化模型 ─────────────────────────────────────────────
# 数据集特征：3016行 / 30列 / 严重不平衡(sick≈6%) / 序列较长(≈300-400 token)
# gpt2-medium: 345M 参数，bf16 约 690MB，RTX5080 16GB 无压力
# GPT-2 注意力投影层名称为 c_attn(合并QKV) / c_proj(输出)，
# 不同于 Qwen 系列的 q_proj/k_proj/v_proj/o_proj
model = GReaT(
    llm=MODEL_ID,
    epochs=50,           # 3016/8≈377步/epoch，50epoch≈18850步，充足
    batch_size=8,        # 30列文本长(≈300-400 token)，batch=8保证显存稳定
    float_precision=3,   # TSH/T3/TT4等连续值保留3位小数
    bf16=True,           # RTX5080 原生支持 bf16
    dataloader_num_workers=0,  # Windows 下保持0，避免多进程问题
    efficient_finetuning="lora",
    lora_config={
        "r": 16,
        "lora_alpha": 32,      # alpha/r = 2，标准比例
        # GPT-2 使用 Conv1D 结构：
        #   c_attn — 单矩阵同时投影 Q/K/V（3*embed_dim）
        #   c_proj — 注意力输出投影
        "target_modules": ["c_attn", "c_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
)

# ── 3. 微调 ───────────────────────────────────────────────────
model.fit(df)

# ── 4. 保存模型（与 Qwen 目录隔离）──────────────────────────────
model_dir = HERE / "sick_great_model" / MODEL_ID
model.save(str(model_dir))
print(f"模型已保存至 {model_dir}")

# ── 5. 生成 3016 个合成样本（与训练集等量，符合论文设置）────────
synthetic_df = model.sample(
    n_samples=3016,
    guided_sampling=True,
    random_feature_order=True,
    max_length=512,      # 30列文本长，512 token 保证完整生成；gpt2-medium 上限1024
    temperature=0.7,
)

print(f"\n合成数据: {synthetic_df.shape[0]} 行, {synthetic_df.shape[1]} 列")
print(synthetic_df.head())
print("\nClass 分布:")
print(synthetic_df["Class"].value_counts())
if "Class" in synthetic_df.columns:
    print(f"正例率 (sick): {synthetic_df['Class'].eq('sick').mean()*100:.2f}%")

# ── 6. 保存合成数据（与 Qwen 目录隔离）──────────────────────────
out_dir = HERE / "data" / MODEL_ID
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "sick_synthetic.csv"
synthetic_df.to_csv(out_path, index=False)
print(f"\n合成数据已保存至 {out_path}")
