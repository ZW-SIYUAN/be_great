"""
Sick (Thyroid) 数据集评估。
复现论文 Tables 1, 2, 4 和 Figure 5（GReaT, tabularisai/Qwen3-0.3B-distil, LoRA）。

用法:
  python evaluation.py                          # 默认 Qwen3-0.3B-distil
  python evaluation.py --model gpt2-medium      # 指定模型

固定数据划分（不引入随机性）：
  data/sick_train.csv        — 80% 真实数据
  data/sick_test.csv         — 20% 真实数据（评估，模型不可见）
  data/<model_id>/sick_synthetic.csv — GReaT 生成的合成数据

Sick 数据集特殊处理（均封装在 load_all() 内）：
  - TBG 列在真实和合成数据中全为空，直接丢弃
  - GReaT 偶尔生成无效 Class 值（other/pregnant/on 等），过滤保留合法类别
  - TSH/T3/TT4/T4U/FTI 列存在缺失值，用训练集中位数填补
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.impute import SimpleImputer

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

HERE          = Path(__file__).parent
TARGET        = "Class"
VALID_CLASSES = {"negative", "sick"}   # GReaT 可能生成 other/pregnant/on 等无效值
DROP_COLS     = ["TBG"]                # 真实和合成数据中全为空，无信息量


# ── Sick 专用清洗（局部函数，不污染 shared/）──────────────────────────────────

def _clean(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """丢弃无意义列，过滤 Class 列中的无效标签。"""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    before = len(df)
    df = df[df[TARGET].isin(VALID_CLASSES)].reset_index(drop=True)
    if (dropped := before - len(df)):
        print(f"[info] {label}: dropped {dropped} malformed row(s) "
              f"({before} -> {len(df)})")
    return df


# ── 数据加载（含 Sick 专用预处理）────────────────────────────────────────────

def load_all(model_id: str = "Qwen3-0.3B-distil"):
    data_dir = HERE / "data"
    syn_path = data_dir / model_id / "sick_synthetic.csv"

    train_real = _clean(pd.read_csv(data_dir / "sick_train.csv"), "train_real")
    test_real  = _clean(pd.read_csv(data_dir / "sick_test.csv"),  "test_real")
    syn_great  = _clean(pd.read_csv(syn_path),                    "syn_great")

    all_dfs  = [train_real, test_real, syn_great]
    cat_cols = [c for c in train_real.columns if train_real[c].dtype == "object"]
    encode_categoricals(all_dfs, cat_cols)   # 包含 TARGET 列

    # 用训练集中位数填补缺失的连续值（TSH/T3/TT4/T4U/FTI）
    feature_cols = [c for c in train_real.columns if c != TARGET]
    num_cols     = [c for c in feature_cols if train_real[c].dtype != "object"]
    imputer      = SimpleImputer(strategy="median")
    imputer.fit(train_real[num_cols])
    for d in all_dfs:
        d[num_cols] = imputer.transform(d[num_cols])

    return train_real, test_real, syn_great, feature_cols


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(model_id: str = "Qwen3-0.3B-distil") -> None:
    out_dir = HERE / "results" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and encoding data ...")
    train_real, test_real, syn_great, feature_cols = load_all(model_id)

    syn_dict  = {"great": syn_great}
    syn_names = list(syn_dict.keys())
    label     = f"Sick — Thyroid ({model_id})"

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
    parser = argparse.ArgumentParser(description="Evaluate GReaT on Sick dataset")
    parser.add_argument("--model", default="Qwen3-0.3B-distil",
                        help="Model ID (subdirectory under data/)")
    args = parser.parse_args()
    main(args.model)
