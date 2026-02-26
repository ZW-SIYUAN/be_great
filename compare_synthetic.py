"""
compare_synthetic.py  —  GReaT vs ChatGPT vs Gemini 三方对比
=============================================================================
固定数据划分：
  normal_experiment/travel/data/travel_train.csv   (763 行, 80%)
  normal_experiment/travel/data/travel_test.csv    (191 行, 20%)
  normal_experiment/travel/data/Qwen3-0.3B-distil/travel_synthetic.csv
  interesting experiment/data/travel_synthetic_chatgpt.csv
  interesting experiment/data/travel_synthetic_gemini.csv

输出（./comparison_results/）：
  Table 1  — ML Utility: Accuracy
  Table 2  — Discriminator Accuracy
  Table 4  — ML Utility: ROCAUC + F1
  Figure 5 — DCR 分布（clipped at 99th pct）
  Figure 6 — Target 类别分布对比
  Figure 7 — 逐列特征分布对比
  Figure 8 — ML Utility 雷达图
  tables.csv / discriminator.csv
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "comparison_results"
OUT_DIR.mkdir(exist_ok=True)
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
    print_table1,
    print_table2,
    print_table4,
)

# ── 路径 ──────────────────────────────────────────────────────────────────────
TRAIN_PATH   = ROOT / "normal_experiment/travel/data/travel_train.csv"
TEST_PATH    = ROOT / "normal_experiment/travel/data/travel_test.csv"
GREAT_PATH   = ROOT / "normal_experiment/travel/data/Qwen3-0.3B-distil/travel_synthetic.csv"
CHATGPT_PATH = ROOT / "interesting_experiment/data/travel_synthetic_chatgpt.csv"
GEMINI_PATH  = ROOT / "interesting_experiment/data/travel_synthetic_gemini.csv"

TARGET       = "Target"
METHOD_NAMES = ["great", "chatgpt", "gemini"]
DISPLAY      = {"original": "Original", "great": "GReaT",
                "chatgpt":  "ChatGPT",  "gemini": "Gemini"}
COLORS       = {"original": "#4C72B0", "great": "#DD8452",
                "chatgpt":  "#55A868", "gemini": "#C44E52"}
SEP          = "=" * 82


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_all():
    train_real = pd.read_csv(TRAIN_PATH)
    test_real  = pd.read_csv(TEST_PATH)
    raws = {
        "great":   pd.read_csv(GREAT_PATH),
        "chatgpt": pd.read_csv(CHATGPT_PATH),
        "gemini":  pd.read_csv(GEMINI_PATH),
    }

    for name in raws:
        before = len(raws[name])
        raws[name] = raws[name].dropna().reset_index(drop=True)
        if (dropped := before - len(raws[name])):
            print(f"  [info] {name}: dropped {dropped} malformed row(s)")

    all_dfs  = [train_real, test_real] + list(raws.values())
    cat_cols = [c for c in train_real.columns if train_real[c].dtype == "object"]
    # encode_categoricals 返回 encoders，供 Figure 7 使用
    encoders = encode_categoricals(all_dfs, cat_cols)

    feature_cols = [c for c in train_real.columns if c != TARGET]
    return train_real, test_real, raws, feature_cols, cat_cols, encoders


# ── 统计保真度（本文件独有）──────────────────────────────────────────────────

def print_fidelity(train_real, raws, feature_cols, cat_cols):
    num_cols = [c for c in feature_cols if c not in cat_cols]
    print(f"\n{SEP}")
    print("STATISTICAL FIDELITY  (vs. train_real)")
    print(f"{'Method':<10}  {'avg KS (num)':>14}  {'avg Wass (num)':>16}  "
          f"{'avg TVD (cat)':>15}")
    print("-" * 62)
    for name, df in raws.items():
        ks_vals, ws_vals, tv_vals = [], [], []
        for col in num_cols:
            ks_vals.append(ks_2samp(train_real[col], df[col]).statistic)
            ws_vals.append(wasserstein_distance(train_real[col], df[col]))
        for col in cat_cols:
            cats = set(train_real[col].unique()) | set(df[col].unique())
            r = train_real[col].value_counts(normalize=True)
            s = df[col].value_counts(normalize=True)
            tv_vals.append(sum(abs(r.get(c, 0) - s.get(c, 0)) for c in cats) / 2)
        print(f"{DISPLAY[name]:<10}  {np.mean(ks_vals):14.4f}  "
              f"{np.mean(ws_vals):16.4f}  {np.mean(tv_vals):15.4f}")
    print(SEP)


# ── Figure 6：Target 类别分布（本文件独有）───────────────────────────────────

def plot_figure6(train_real, raws):
    all_data = {"Original\n(train)": train_real}
    all_data.update({DISPLAY[n] + "\n(syn)": df for n, df in raws.items()})

    labels   = list(all_data.keys())
    pos_rate = [df[TARGET].mean() * 100 for df in all_data.values()]
    neg_rate = [100 - p for p in pos_rate]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, neg_rate, color="#AEC6E8", label="Target=0 (No Insurance)")
    ax.bar(x, pos_rate, bottom=neg_rate, color="#F4A460",
           label="Target=1 (Bought Insurance)")

    for i, (p, n) in enumerate(zip(pos_rate, neg_rate)):
        ax.text(i, n + p/2, f"{p:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(i, n/2, f"{100-p:.1f}%", ha="center", va="center",
                fontsize=9, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Figure 6: Target Class Distribution — Real vs Synthetic")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure6_target_dist.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Figure 7：逐列特征分布（本文件独有）─────────────────────────────────────

def plot_figure7(train_real_raw, raws_raw, feature_cols, cat_cols):
    """使用未编码（原始字符串）的 DataFrame，保证坐标轴可读。"""
    num_cols  = [c for c in feature_cols if c not in cat_cols]
    all_cols  = num_cols + cat_cols
    ncols_fig = 3
    nrows_fig = int(np.ceil(len(all_cols) / ncols_fig))

    all_src    = {"Original": train_real_raw}
    all_src.update({DISPLAY[n]: df for n, df in raws_raw.items()})
    src_colors = {DISPLAY[n]: COLORS[n] for n in METHOD_NAMES}
    src_colors["Original"] = COLORS["original"]

    fig, axes = plt.subplots(nrows_fig, ncols_fig,
                             figsize=(5 * ncols_fig, 4 * nrows_fig))
    axes = axes.flatten()

    for i, col in enumerate(all_cols):
        ax = axes[i]
        if col in num_cols:
            all_vals = pd.concat([df[col] for df in all_src.values()])
            bins = np.linspace(all_vals.min(), all_vals.max(), 20)
            for src_name, df in all_src.items():
                ax.hist(df[col], bins=bins, alpha=0.55, density=True,
                        color=src_colors[src_name], label=src_name)
        else:
            cats = sorted(train_real_raw[col].unique().tolist())
            x    = np.arange(len(cats))
            w    = 0.8 / len(all_src)
            for j, (src_name, df) in enumerate(all_src.items()):
                vc   = df[col].value_counts(normalize=True)
                vals = [vc.get(c, 0) for c in cats]
                ax.bar(x + (j - len(all_src)/2 + 0.5)*w, vals, w,
                       color=src_colors[src_name], alpha=0.80, label=src_name)
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in cats],
                               rotation=20, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 7: Column-wise Distributions — Real vs All Synthetic",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure7_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Figure 8：ML Utility 雷达图（本文件独有）────────────────────────────────

def plot_figure8(store):
    metrics   = ["acc", "auc", "f1"]
    m_labels  = ["Accuracy", "ROCAUC", "F1"]
    clf_names = ["LR", "DT", "RF"]
    angles    = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles   += angles[:1]

    orig_vals = {m: {k: np.mean(store[m]["original"][k]) for k in metrics}
                 for m in clf_names}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(polar=True))
    for ax, clf in zip(axes, clf_names):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), m_labels, fontsize=10)
        ax.set_ylim(50, 100)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.set_yticklabels(["60", "70", "80", "90", "100"], fontsize=7)

        orig = [orig_vals[clf][k] for k in metrics] + [orig_vals[clf][metrics[0]]]
        ax.plot(angles, orig, "k--", lw=1.5, label="Original")
        ax.fill(angles, orig, alpha=0.05, color="black")

        for name in METHOD_NAMES:
            vals = [np.mean(store[clf][name][k]) for k in metrics] + \
                   [np.mean(store[clf][name][metrics[0]])]
            ax.plot(angles, vals, lw=2, color=COLORS[name], label=DISPLAY[name])
            ax.fill(angles, vals, alpha=0.10, color=COLORS[name])

        ax.set_title(f"Classifier: {clf}", fontsize=11, pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    fig.suptitle("Figure 8: ML Utility Radar — GReaT vs ChatGPT vs Gemini "
                 "(Train on Synthetic, Test on Real)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure8_radar.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 保存 CSV（长格式，与 normal_experiment 的宽格式不同）─────────────────────

def save_csvs(store, disc):
    rows = []
    for m in ["LR", "DT", "RF"]:
        for split in ["original"] + METHOD_NAMES:
            for metric in ["acc", "f1", "auc"]:
                vals = np.array(store[m][split][metric])
                rows.append({"classifier": m, "method": split,
                              "metric": metric,
                              "mean": round(vals.mean(), 4),
                              "std":  round(vals.std(),  4)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "tables.csv", index=False)

    pd.DataFrame([
        {"method": n,
         "disc_acc_mean": round(np.mean(v), 4),
         "disc_acc_std":  round(np.std(v),  4)}
        for n, v in disc.items()
    ]).to_csv(OUT_DIR / "discriminator.csv", index=False)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    train_real, test_real, raws, feature_cols, cat_cols, encoders = load_all()

    # 保留原始未编码副本，供 Figure 7 使用
    train_raw = pd.read_csv(TRAIN_PATH)
    raws_raw  = {
        "great":   pd.read_csv(GREAT_PATH).dropna().reset_index(drop=True),
        "chatgpt": pd.read_csv(CHATGPT_PATH),
        "gemini":  pd.read_csv(GEMINI_PATH),
    }

    # 数据概览
    print(f"\n{SEP}")
    print("DATASET OVERVIEW")
    print(SEP)
    for label, df in [("train_real", train_real), ("test_real", test_real)] + \
                     [(n, d) for n, d in raws.items()]:
        pos = df[TARGET].mean() * 100
        print(f"  {label:<12}: {len(df):5d} rows  "
              f"Target={{0:{int(len(df)*(1-pos/100))}, 1:{int(len(df)*pos/100)}}}  "
              f"pos_rate={pos:.1f}%")
    print(SEP)

    # 统计保真度
    print_fidelity(train_real, raws, feature_cols, cat_cols)

    # ML 效用（Tables 1 & 4）
    print(f"\nRunning ML utility (5 trials × 3 models) ...")
    store = compute_ml_tables(train_real, test_real, raws, feature_cols, TARGET)
    print_table1(store, METHOD_NAMES,
                 dataset_name="Travel", display_names=DISPLAY)
    print_table4(store, METHOD_NAMES,
                 dataset_name="Travel", display_names=DISPLAY)

    # 判别器（Table 2）
    print(f"\nRunning discriminator (5 trials × 3 methods) ...")
    disc = compute_discriminator(train_real, raws, feature_cols)
    print_table2(disc, display_names=DISPLAY)

    # DCR（Figure 5）
    print(f"\nComputing DCR ...")
    dcr_test, dcr_syn = compute_dcr(train_real, test_real, raws, feature_cols)
    print_dcr_summary(dcr_test, dcr_syn, display_names=DISPLAY)
    plot_dcr(dcr_test, dcr_syn,
             out_path=OUT_DIR / "figure5_dcr.png",
             dataset_name="Travel",
             colors=COLORS,
             display_names=DISPLAY,
             clip_p99=True)

    # Figures 6 / 7 / 8（本文件独有的可视化）
    print(f"\nGenerating figures ...")
    plot_figure6(train_real, raws)
    plot_figure7(train_raw, raws_raw, feature_cols, cat_cols)
    plot_figure8(store)

    # 保存 CSV
    save_csvs(store, disc)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
