"""
Travel 数据集：ChatGPT vs Gemini 对比评估。
复现论文 Tables 1, 2, 4 和 Figure 5。

固定数据划分（不引入随机性）：
  data/travel_train.csv  — 80% 真实数据
  data/travel_test.csv   — 20% 真实数据（评估）
  data/travel_synthetic_chatgpt.csv — ChatGPT 生成
  data/travel_synthetic_gemini.csv  — Gemini 生成
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

ROOT = Path(__file__).parents[1]
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

HERE    = Path(__file__).parent
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

TARGET = "Target"

DISPLAY = {"chatgpt": "ChatGPT", "gemini": "Gemini"}
COLORS  = {"chatgpt": "darkorange", "gemini": "green"}


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_all():
    data_dir   = HERE / "data"
    train_real = pd.read_csv(data_dir / "travel_train.csv")
    test_real  = pd.read_csv(data_dir / "travel_test.csv")
    syn_cg     = pd.read_csv(data_dir / "travel_synthetic_chatgpt.csv")
    syn_ge     = pd.read_csv(data_dir / "travel_synthetic_gemini.csv")

    all_dfs  = [train_real, test_real, syn_cg, syn_ge]
    cat_cols = [c for c in train_real.columns if train_real[c].dtype == "object"]
    encode_categoricals(all_dfs, cat_cols)

    feature_cols = [c for c in train_real.columns if c != TARGET]
    return train_real, test_real, syn_cg, syn_ge, feature_cols


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("Loading and encoding data ...")
    train_real, test_real, syn_cg, syn_ge, feature_cols = load_all()

    syn_dict  = {"chatgpt": syn_cg, "gemini": syn_ge}
    syn_names = list(syn_dict.keys())

    print_overview(train_real, test_real, syn_dict, TARGET)

    # Table 1 & 4
    print(f"\nRunning ML utility (5 trials × 3 models × {len(syn_names)+1} splits) ...")
    store = compute_ml_tables(train_real, test_real, syn_dict, feature_cols, TARGET)
    print_table1(store, syn_names,
                 dataset_name="Travel (ChatGPT vs Gemini)",
                 display_names=DISPLAY)
    print_table4(store, syn_names,
                 dataset_name="Travel (ChatGPT vs Gemini)",
                 display_names=DISPLAY)

    # Table 2
    print(f"\nRunning discriminator (5 trials) ...")
    disc = compute_discriminator(train_real, syn_dict, feature_cols)
    print_table2(disc, display_names=DISPLAY)

    # Figure 5
    print(f"\nComputing DCR distributions ...")
    dcr_test, dcr_syn = compute_dcr(train_real, test_real, syn_dict, feature_cols)
    print_dcr_summary(dcr_test, dcr_syn, display_names=DISPLAY)
    plot_dcr(dcr_test, dcr_syn,
             out_path=OUT_DIR / "figure5_dcr.png",
             dataset_name="Travel (ChatGPT vs Gemini)",
             colors=COLORS,
             display_names=DISPLAY)

    # 保存 CSV
    save_results(store, disc, syn_names, OUT_DIR)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
