"""
California Housing 数据集训练脚本。

用法:
  python train.py --config configs/qwen3.yaml

数据集特征：
  - 16512 行 / 10 列（9 个连续值 + 1 个分类列 ocean_proximity）
  - 目标列 median_house_value 为连续回归值（GReaT 以文本形式生成，无需特殊处理）
  - total_bedrooms 列含 174 个空值（< 1%），训练前 dropna 清除
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from be_great import GReaT

HERE = Path(__file__).parent


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    train_csv = HERE / cfg["dataset"]["train_csv"]
    model_dir = HERE / cfg["output"]["model_dir"]
    syn_out   = HERE / cfg["output"]["synthetic_csv"]
    model_dir.mkdir(parents=True, exist_ok=True)
    syn_out.parent.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    df = pd.read_csv(train_csv)
    print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

    # total_bedrooms 有 174 个空值（< 1%），直接丢弃
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if (dropped := before - len(df)):
        print(f"[info] Dropped {dropped} rows with NaN ({before} -> {len(df)})")

    target = cfg["dataset"]["target_col"]
    print(f"\n训练数据: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"\n{target} 统计:\n{df[target].describe().round(2)}")
    print(f"\nocean_proximity 分布:\n{df['ocean_proximity'].value_counts()}")

    # ── 初始化模型 ────────────────────────────────────────────────────────────
    model = GReaT(
        llm=cfg["model"]["llm"],
        epochs=cfg["model"]["epochs"],
        batch_size=cfg["model"]["batch_size"],
        float_precision=cfg["model"]["float_precision"],
        bf16=cfg["model"]["bf16"],
        dataloader_num_workers=cfg["model"]["dataloader_num_workers"],
        efficient_finetuning="lora",
        lora_config=cfg["lora"],
    )

    # ── 微调 ──────────────────────────────────────────────────────────────────
    model.fit(df)

    # ── 保存模型 ──────────────────────────────────────────────────────────────
    model.save(str(model_dir))
    print(f"\n模型已保存至 {model_dir}")

    # ── 生成合成数据 ──────────────────────────────────────────────────────────
    s = cfg["sampling"]
    synthetic_df = model.sample(
        n_samples=cfg["dataset"]["n_samples"],
        guided_sampling=s["guided_sampling"],
        random_feature_order=s["random_feature_order"],
        max_length=cfg["dataset"]["max_length"],
        temperature=s["temperature"],
    )

    print(f"\n合成数据: {synthetic_df.shape[0]} 行, {synthetic_df.shape[1]} 列")
    if target in synthetic_df.columns:
        print(f"\n{target} 统计:\n{synthetic_df[target].describe().round(2)}")
    if "ocean_proximity" in synthetic_df.columns:
        print(f"\nocean_proximity 分布:\n{synthetic_df['ocean_proximity'].value_counts()}")

    # ── 保存合成数据 ──────────────────────────────────────────────────────────
    synthetic_df.to_csv(syn_out, index=False)
    print(f"\n合成数据已保存至 {syn_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GReaT 训练 — California Housing")
    parser.add_argument("--config", required=True,
                        help="YAML 配置文件路径（如 configs/qwen3.yaml）")
    args = parser.parse_args()
    main(args.config)
