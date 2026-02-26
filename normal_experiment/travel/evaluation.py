"""
Travel 数据集评估。
复现论文 Tables 1, 2, 4 和 Figure 5（GReaT, tabularisai/Qwen3-0.3B-distil, LoRA）。

用法:
  python evaluation.py                          # 默认 Qwen3-0.3B-distil
  python evaluation.py --model gpt2-medium      # 指定模型

固定数据划分（不引入随机性）：
  data/travel_train.csv        — 80% 真实数据（微调 GReaT 时使用）
  data/travel_test.csv         — 20% 真实数据（评估，模型不可见）
  data/<model_id>/travel_synthetic.csv — GReaT 生成的合成数据
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

# ── 路径设置：把 repo 根目录加入 sys.path，以便 import shared/ ─────────────────
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from shared.eval_core import (
    compute_dcr,
    compute_discriminator,
    compute_ml_tables,
    encode_categoricals,
)
from shared.eval_tables import (
    plot_dcr,
    print_dcr_summary,
    print_overview,
    print_table1,
    print_table2,
    print_table4,
    save_results,
)

HERE   = Path(__file__).parent
TARGET = "Target"


# ── 数据加载（Travel 无特殊预处理）────────────────────────────────────────────

def load_all(model_id: str = "Qwen3-0.3B-distil"):
    data_dir = HERE / "data"
    syn_path = data_dir / model_id / "travel_synthetic.csv"

    train_real = pd.read_csv(data_dir / "travel_train.csv")
    test_real  = pd.read_csv(data_dir / "travel_test.csv")
    syn_great  = pd.read_csv(syn_path)

    # GReaT 偶尔生成格式错误的行（列值溢出），dropna 清除
    before = len(syn_great)
    syn_great = syn_great.dropna().reset_index(drop=True)
    if (dropped := before - len(syn_great)):
        print(f"[info] Dropped {dropped} malformed row(s) from synthetic data "
              f"({before} -> {len(syn_great)})")

    all_dfs  = [train_real, test_real, syn_great]
    cat_cols = [c for c in train_real.columns if train_real[c].dtype == "object"]
    encode_categoricals(all_dfs, cat_cols)

    feature_cols = [c for c in train_real.columns if c != TARGET]
    return train_real, test_real, syn_great, feature_cols


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(model_id: str = "Qwen3-0.3B-distil") -> None:
    out_dir = HERE / "results" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and encoding data ...")
    train_real, test_real, syn_great, feature_cols = load_all(model_id)

    syn_dict  = {"great": syn_great}
    syn_names = list(syn_dict.keys())
    label     = f"Travel ({model_id})"

    print_overview(train_real, test_real, syn_dict, TARGET)

    # Table 1 & 4
    print(f"\nRunning ML utility (5 trials × 3 models × {len(syn_names)+1} splits) ...")
    store = compute_ml_tables(train_real, test_real, syn_dict, feature_cols, TARGET)
    print_table1(store, syn_names, dataset_name=label)
    print_table4(store, syn_names, dataset_name=label)

    # Table 2
    print(f"\nRunning discriminator (5 trials) ...")
    disc = compute_discriminator(train_real, syn_dict, feature_cols)
    print_table2(disc)

    # Figure 5 + DCR 摘要
    print(f"\nComputing DCR distributions ...")
    dcr_test, dcr_syn = compute_dcr(train_real, test_real, syn_dict, feature_cols)
    print_dcr_summary(dcr_test, dcr_syn)
    plot_dcr(dcr_test, dcr_syn,
             out_path=out_dir / "figure5_dcr.png",
             dataset_name=label)

    # 保存 CSV
    save_results(store, disc, syn_names, out_dir)
    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GReaT on Travel dataset")
    parser.add_argument("--model", default="Qwen3-0.3B-distil",
                        help="Model ID (subdirectory under data/)")
    args = parser.parse_args()
    main(args.model)
