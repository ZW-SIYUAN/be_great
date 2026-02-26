"""
shared/eval_tables.py
=====================
打印表格、绘制图形、保存 CSV。
所有函数接受 dataset_name / display_names 参数以适配不同实验。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEP = "=" * 76


# ── 格式化工具 ────────────────────────────────────────────────────────────────

def _ms(vals) -> str:
    """格式化 mean±std 字符串，宽度对齐。"""
    a = np.array(vals)
    return f"{a.mean():6.2f}+-{a.std():.2f}"


# ── 打印表格 ──────────────────────────────────────────────────────────────────

def print_overview(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    syn_dict: Dict,
    target: str,
) -> None:
    """打印数据集概览：行数和目标分布。"""
    print(f"\n{SEP}")
    print("DATASET OVERVIEW")
    print(SEP)

    def tgt(df):
        return df[target].value_counts().sort_index().to_dict()

    print(f"  train_real  : {len(train_real):5d} rows  Target={tgt(train_real)}")
    print(f"  test_real   : {len(test_real):5d} rows  Target={tgt(test_real)}")
    for name, df in syn_dict.items():
        print(f"  {name:<12}: {len(df):5d} rows  Target={tgt(df)}")
    print(SEP)


def print_table1(
    store: Dict,
    syn_names: List[str],
    dataset_name: str = "",
    display_names: Optional[Dict[str, str]] = None,
) -> None:
    """
    Table 1: ML Utility — Accuracy (%)
    display_names: {key: "显示名"} 用于列标题，默认 capitalize()。
    """
    display_names = display_names or {}

    def label(n):
        return display_names.get(n, n.capitalize())

    print(f"\n{SEP}")
    print(f"TABLE 1  ML Utility — Accuracy (%)  |  Dataset: {dataset_name}")
    header = f"{'Model':<6}  {'Original':>16}" + \
             "".join(f"  {label(n):>16}" for n in syn_names) + \
             "".join(f"  {'d_'+n[:2]:>8}" for n in syn_names)
    print(header)
    print("-" * len(header))
    for m in ["LR", "DT", "RF"]:
        orig = np.mean(store[m]["original"]["acc"])
        line = f"{m:<6}  {_ms(store[m]['original']['acc'])}"
        for name in syn_names:
            line += f"  {_ms(store[m][name]['acc'])}"
        for name in syn_names:
            line += f"  {np.mean(store[m][name]['acc']) - orig:+7.2f}"
        print(line)
    print(SEP)


def print_table2(
    disc: Dict[str, List[float]],
    display_names: Optional[Dict[str, str]] = None,
) -> None:
    """
    Table 2: Discriminator Accuracy (%)
    越低越好，50% = 完全无法区分。
    """
    display_names = display_names or {}
    print(f"\n{SEP}")
    print("TABLE 2  Discriminator Accuracy (%)  |  lower=better, 50%=indistinguishable")
    print(SEP)
    for name, vals in disc.items():
        a     = np.array(vals)
        label = display_names.get(name, name.capitalize())
        print(f"  {label:<12} {a.mean():.2f}+-{a.std():.2f}  "
              f"trials={[round(v, 2) for v in vals]}")
    print(SEP)


def print_table4(
    store: Dict,
    syn_names: List[str],
    dataset_name: str = "",
    display_names: Optional[Dict[str, str]] = None,
) -> None:
    """Table 4: ML Utility — ROCAUC (%) and F1 (%)"""
    display_names = display_names or {}

    def label(n):
        return display_names.get(n, n.capitalize())

    print(f"\n{SEP}")
    print(f"TABLE 4  ML Utility — ROCAUC (%) and F1 (%)  |  Dataset: {dataset_name}")
    header = f"{'Model':<6}" + \
             "".join(f"  {label(n)+' AUC':>16}  {label(n)+' F1':>16}"
                     for n in syn_names)
    print(header)
    print("-" * len(header))
    for m in ["LR", "DT", "RF"]:
        line = f"{m:<6}"
        for name in syn_names:
            line += f"  {_ms(store[m][name]['auc'])}  {_ms(store[m][name]['f1'])}"
        print(line)
    print(SEP)


def print_dcr_summary(
    dcr_test: "np.ndarray",
    dcr_syn: Dict[str, "np.ndarray"],
    display_names: Optional[Dict[str, str]] = None,
) -> None:
    """打印 DCR 均值/中位数/ratio 摘要。"""
    display_names = display_names or {}
    print("\nDCR summary:")
    print(f"  Test   -> Train : mean={dcr_test.mean():.4f}  "
          f"median={np.median(dcr_test):.4f}")
    for name, dcr in dcr_syn.items():
        label = display_names.get(name, name)
        ratio = dcr.mean() / (dcr_test.mean() + 1e-10)
        print(f"  {label:<12}-> Train : mean={dcr.mean():.4f}  "
              f"median={np.median(dcr):.4f}  ratio={ratio:.3f}")


# ── 保存 CSV ──────────────────────────────────────────────────────────────────

def save_results(
    store: Dict,
    disc: Dict,
    syn_names: List[str],
    out_dir: Path,
) -> None:
    """
    保存宽格式 ML utility（table1_table4.csv）和判别器结果（table2.csv）。
    """
    rows = []
    for m in ["LR", "DT", "RF"]:
        row = {"Model": m}
        for split in ["original"] + syn_names:
            for metric in ["acc", "f1", "auc"]:
                vals = np.array(store[m][split][metric])
                row[f"{split}_{metric}_mean"] = round(vals.mean(), 4)
                row[f"{split}_{metric}_std"]  = round(vals.std(),  4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "table1_table4.csv", index=False)

    disc_rows = []
    for name, vals in disc.items():
        a = np.array(vals)
        disc_rows.append({
            "method":         name,
            "disc_acc_mean":  round(a.mean(), 4),
            "disc_acc_std":   round(a.std(),  4),
            "trials":         vals,
        })
    pd.DataFrame(disc_rows).to_csv(out_dir / "table2.csv", index=False)


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_dcr(
    dcr_test: "np.ndarray",
    dcr_syn: Dict[str, "np.ndarray"],
    out_path: Path,
    dataset_name: str = "",
    colors: Optional[Dict[str, str]] = None,
    display_names: Optional[Dict[str, str]] = None,
    clip_p99: bool = False,
) -> None:
    """
    Figure 5: DCR 分布直方图。
    clip_p99=True 时裁剪到 99 百分位（适合多方法对比时的可读性）。
    """
    colors        = colors or {}
    display_names = display_names or {}
    all_vals      = [dcr_test] + list(dcr_syn.values())

    if clip_p99:
        p99 = max(np.percentile(v, 99) for v in all_vals)
        hi  = p99 * 1.05
    else:
        hi = max(v.max() for v in all_vals) * 1.05
    bins = np.linspace(0, hi, 55)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dcr_test, bins=bins, alpha=0.50, color="steelblue", density=True,
            label=f"Original Test Set  (mean={dcr_test.mean():.3f})")

    for name, dcr in dcr_syn.items():
        d_plot = dcr[dcr <= hi] if clip_p99 else dcr
        label  = display_names.get(name, name.capitalize())
        color  = colors.get(name, "darkorange")
        ratio  = dcr.mean() / (dcr_test.mean() + 1e-10)
        ax.hist(d_plot, bins=bins, alpha=0.50, color=color, density=True,
                label=f"{label} Synthetic  (mean={dcr.mean():.3f}, ratio={ratio:.2f}x)")

    ax.set_xlabel("Distance to Closest Record (L2, normalised)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    title_suffix = " (clipped at 99th pct)" if clip_p99 else ""
    ax.set_title(f"Figure 5: DCR Distributions — {dataset_name}\n"
                 f"(Synthetic Data vs. Original Train Set{title_suffix})", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
