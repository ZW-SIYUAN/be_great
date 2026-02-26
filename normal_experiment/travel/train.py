"""
统一训练脚本：从 YAML 读取所有配置，不硬编码任何参数。

用法:
  python train.py --config configs/qwen3.yaml
  python train.py --config configs/gpt2_medium.yaml

添加新模型只需新建一个 YAML 文件，不需要复制脚本。
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from be_great import GReaT

HERE = Path(__file__).parent


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    # 所有路径相对于本脚本所在目录（normal_experiment/travel/）
    train_csv = HERE / cfg["dataset"]["train_csv"]
    model_dir = HERE / cfg["output"]["model_dir"]
    syn_out   = HERE / cfg["output"]["synthetic_csv"]
    model_dir.mkdir(parents=True, exist_ok=True)
    syn_out.parent.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    df     = pd.read_csv(train_csv)
    target = cfg["dataset"]["target_col"]
    print(f"训练数据: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"\n{target} 分布:\n{df[target].value_counts()}")

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

    print(f"\n合成数据: {synthetic_df.shape[0]} 行")
    print(f"\n{target} 分布:\n{synthetic_df[target].value_counts()}")

    # ── 保存合成数据 ──────────────────────────────────────────────────────────
    synthetic_df.to_csv(syn_out, index=False)
    print(f"\n合成数据已保存至 {syn_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GReaT 训练脚本（配置驱动）"
    )
    parser.add_argument(
        "--config", required=True,
        help="YAML 配置文件路径（相对于本脚本目录，如 configs/qwen3.yaml）",
    )
    args = parser.parse_args()
    main(args.config)
