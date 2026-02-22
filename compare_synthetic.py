"""
compare_synthetic.py  —  root-level comparison of three synthetic datasets
===========================================================================
Synthetic methods compared:
  GReaT    — tabularisai/Qwen3-0.3B-distil, LoRA, fine-tuned on travel_train
  ChatGPT  — GPT-4o prompted with travel_train schema + samples
  Gemini   — Gemini prompted with travel_train schema + samples

Fixed data splits:
  travel_train.csv  (763 rows, 80% of original) — used by all generators
  travel_test.csv   (191 rows, 20% of original) — evaluation only, never seen by generators

Outputs  (saved to  ./comparison_results/):
  Table 1  — ML Utility: Accuracy (%)          | LR / DT / RF | Original / GReaT / ChatGPT / Gemini
  Table 2  — Discriminator Accuracy (%)        | RF           | all methods (lower=better)
  Table 4  — ML Utility: ROCAUC + F1 (%)       | LR / DT / RF | all synthetic methods
  Figure 5 — DCR distributions                 | all methods vs Train
  Figure 6 — Target-class distribution bar     | real vs all methods
  Figure 7 — Per-column distribution grid      | real vs all methods
  Figure 8 — TSTR radar chart                  | Acc / AUC / F1 across models
  tables.csv / discriminator.csv               — numeric results

5 trials: fixed data, varying downstream-classifier random seed only.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, ks_2samp

from sklearn.base import clone as sklearn_clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "comparison_results"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH   = ROOT / "normal_experiment/travel/data/travel_train.csv"
TEST_PATH    = ROOT / "normal_experiment/travel/data/travel_test.csv"
GREAT_PATH   = ROOT / "normal_experiment/travel/data/travel_synthetic.csv"
CHATGPT_PATH = ROOT / "interesting experiment/data/travel_synthetic_chatgpt.csv"
GEMINI_PATH  = ROOT / "interesting experiment/data/travel_synthetic_gemini.csv"

TARGET = "Target"
SEEDS  = [42, 1, 2, 3, 4]
DISC_TEST_SIZE = 0.30

# Display names and colours
METHOD_NAMES   = ["great", "chatgpt", "gemini"]
DISPLAY        = {"original": "Original", "great": "GReaT",
                  "chatgpt": "ChatGPT",  "gemini": "Gemini"}
COLORS         = {"original": "#4C72B0", "great": "#DD8452",
                  "chatgpt":  "#55A868", "gemini": "#C44E52"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    train_real = pd.read_csv(TRAIN_PATH)
    test_real  = pd.read_csv(TEST_PATH)
    raws = {
        "great":   pd.read_csv(GREAT_PATH),
        "chatgpt": pd.read_csv(CHATGPT_PATH),
        "gemini":  pd.read_csv(GEMINI_PATH),
    }

    # Drop malformed (NaN) rows from any synthetic set
    for name in raws:
        before = len(raws[name])
        raws[name] = raws[name].dropna().reset_index(drop=True)
        dropped = before - len(raws[name])
        if dropped:
            print(f"  [info] {name}: dropped {dropped} malformed row(s) "
                  f"({before} -> {len(raws[name])})")

    # Encode categoricals over union of all datasets
    all_dfs  = [train_real, test_real] + list(raws.values())
    cat_cols = [c for c in train_real.columns if train_real[c].dtype == "object"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([d[col] for d in all_dfs]).astype(str)
        le.fit(combined)
        for d in all_dfs:
            d[col] = le.transform(d[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in train_real.columns if c != TARGET]
    return train_real, test_real, raws, feature_cols, cat_cols, encoders


# ══════════════════════════════════════════════════════════════════════════════
# 2. Classifiers
# ══════════════════════════════════════════════════════════════════════════════

def get_classifiers(seed):
    return {
        "LR": Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000,
                                                   random_state=seed))]),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed,
                                     n_jobs=-1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. ML Utility  (Tables 1 & 4)
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_trial(train_real, test_real, raws, feature_cols, seed):
    X_te = test_real[feature_cols].values
    y_te = test_real[TARGET].values
    splits = {"original": (train_real[feature_cols].values,
                           train_real[TARGET].values)}
    splits.update({n: (df[feature_cols].values, df[TARGET].values)
                   for n, df in raws.items()})
    results = {}
    for mname, clf in get_classifiers(seed).items():
        row = {}
        for sname, (Xtr, ytr) in splits.items():
            c = sklearn_clone(clf)
            c.fit(Xtr, ytr)
            yp  = c.predict(X_te)
            ypr = c.predict_proba(X_te)[:, 1]
            row[sname] = {
                "acc": accuracy_score(y_te, yp) * 100,
                "f1":  f1_score(y_te, yp, average="weighted",
                                zero_division=0) * 100,
                "auc": roc_auc_score(y_te, ypr) * 100,
            }
        results[mname] = row
    return results


def compute_ml(train_real, test_real, raws, feature_cols):
    models     = ["LR", "DT", "RF"]
    split_keys = ["original"] + METHOD_NAMES
    metrics    = ["acc", "f1", "auc"]
    store = {m: {s: {k: [] for k in metrics} for s in split_keys}
             for m in models}
    for seed in SEEDS:
        trial = run_ml_trial(train_real, test_real, raws, feature_cols, seed)
        for m in models:
            for s in split_keys:
                for k in metrics:
                    store[m][s][k].append(trial[m][s][k])
    return store


# ══════════════════════════════════════════════════════════════════════════════
# 4. Discriminator  (Table 2)
# ══════════════════════════════════════════════════════════════════════════════

def discriminator_trial(train_real, syn_df, feature_cols, seed):
    X = np.vstack([train_real[feature_cols].values,
                   syn_df[feature_cols].values])
    y = np.array([1]*len(train_real) + [0]*len(syn_df))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=DISC_TEST_SIZE, stratify=y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)) * 100


def compute_discriminator(train_real, raws, feature_cols):
    return {name: [discriminator_trial(train_real, df, feature_cols, s)
                   for s in SEEDS]
            for name, df in raws.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Print tables
# ══════════════════════════════════════════════════════════════════════════════

SEP = "=" * 82

def ms(vals):
    a = np.array(vals)
    return f"{a.mean():6.2f}+-{a.std():.2f}"


def print_table1(store):
    print(f"\n{SEP}")
    print("TABLE 1  ML Utility — Accuracy (%)  |  Dataset: Travel  |  Test on real")
    hdr = f"{'Model':<6}  {'Original':>15}" + \
          "".join(f"  {DISPLAY[n]:>15}" for n in METHOD_NAMES) + \
          "".join(f"  {'d_'+n[:3]:>7}" for n in METHOD_NAMES)
    print(hdr); print("-" * len(hdr))
    for m in ["LR", "DT", "RF"]:
        orig = np.mean(store[m]["original"]["acc"])
        line = f"{m:<6}  {ms(store[m]['original']['acc'])}"
        for n in METHOD_NAMES:
            line += f"  {ms(store[m][n]['acc'])}"
        for n in METHOD_NAMES:
            line += f"  {np.mean(store[m][n]['acc'])-orig:+6.2f}"
        print(line)
    print(SEP)


def print_table2(disc):
    print(f"\n{SEP}")
    print("TABLE 2  Discriminator Accuracy (%)  |  lower=better  |  50%=indistinguishable")
    print(SEP)
    for name, vals in disc.items():
        a = np.array(vals)
        bar = "#" * int(a.mean() - 50)
        print(f"  {DISPLAY[name]:<10}  {a.mean():5.2f}+-{a.std():.2f}  [{bar}]")
    print(SEP)


def print_table4(store):
    print(f"\n{SEP}")
    print("TABLE 4  ML Utility — ROCAUC (%) and F1 (%)  |  Dataset: Travel")
    hdr = f"{'Model':<6}" + \
          "".join(f"  {DISPLAY[n]+' AUC':>14}  {DISPLAY[n]+' F1':>14}"
                  for n in METHOD_NAMES)
    print(hdr); print("-" * len(hdr))
    for m in ["LR", "DT", "RF"]:
        line = f"{m:<6}"
        for n in METHOD_NAMES:
            line += f"  {ms(store[m][n]['auc'])}  {ms(store[m][n]['f1'])}"
        print(line)
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Figure 5 — DCR distributions
# ══════════════════════════════════════════════════════════════════════════════

def compute_dcr(train_real, test_real, raws, feature_cols):
    X_train = train_real[feature_cols].values.astype(float)
    X_test  = test_real[feature_cols].values.astype(float)
    col_min = X_train.min(0); col_max = X_train.max(0)
    rng = np.where(col_max > col_min, col_max - col_min, 1.0)
    Xtr_n = (X_train - col_min) / rng
    Xte_n = (X_test  - col_min) / rng
    dcr_test = cdist(Xte_n, Xtr_n, metric="euclidean").min(axis=1)
    dcr_syn  = {n: cdist((df[feature_cols].values.astype(float)-col_min)/rng,
                         Xtr_n, metric="euclidean").min(axis=1)
                for n, df in raws.items()}
    return dcr_test, dcr_syn


def plot_figure5(dcr_test, dcr_syn):
    # Clip extreme outliers at 99th percentile for readability
    p99 = max(np.percentile(v, 99) for v in list(dcr_syn.values()) + [dcr_test])
    bins = np.linspace(0, p99 * 1.05, 50)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dcr_test, bins=bins, density=True, alpha=0.50,
            color=COLORS["original"],
            label=f"Original Test  (mean={dcr_test.mean():.3f})")
    for name, dcr in dcr_syn.items():
        d_clip = dcr[dcr <= p99 * 1.05]
        ax.hist(d_clip, bins=bins, density=True, alpha=0.50,
                color=COLORS[name],
                label=f"{DISPLAY[name]} Synthetic  "
                      f"(mean={dcr.mean():.3f}, ratio={dcr.mean()/dcr_test.mean():.2f}x)")

    ax.set_xlabel("Distance to Closest Record (L2, normalised)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Figure 5: DCR Distributions — Travel Dataset\n"
                 "(Synthetic vs. Original Train Set, clipped at 99th pct)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure5_dcr.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Figure 6 — Target distribution bar
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure6(train_real, raws):
    all_data = {"Original\n(train)": train_real}
    all_data.update({DISPLAY[n]+"\n(syn)": df for n, df in raws.items()})

    labels   = list(all_data.keys())
    pos_rate = [df[TARGET].mean() * 100 for df in all_data.values()]
    neg_rate = [100 - p for p in pos_rate]
    clr_pos  = [COLORS.get(k.split("\n")[0].lower().replace(" ","").replace("(syn)","").replace("(train)",""),
                           "#4C72B0") for k in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars_neg = ax.bar(x, neg_rate, color="#AEC6E8", label="Target=0 (No Insurance)")
    bars_pos = ax.bar(x, pos_rate, bottom=neg_rate, color="#F4A460",
                      label="Target=1 (Bought Insurance)")

    for i, (p, n) in enumerate(zip(pos_rate, neg_rate)):
        ax.text(i, n + p/2, f"{p:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(i, n/2, f"{100-p:.1f}%", ha="center", va="center",
                fontsize=9, color="#333")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Figure 6: Target Class Distribution — Real vs Synthetic")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure6_target_dist.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 8. Figure 7 — Per-column distribution grid
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure7(train_real_raw, raws_raw, feature_cols, cat_cols, encoders):
    """Uses un-encoded (raw string) dataframes for readable labels."""
    num_cols = [c for c in feature_cols if c not in cat_cols]
    all_cols = num_cols + cat_cols
    ncols_fig = 3
    nrows_fig = int(np.ceil(len(all_cols) / ncols_fig))

    all_src  = {"Original": train_real_raw}
    all_src.update({DISPLAY[n]: df for n, df in raws_raw.items()})
    src_colors = {"Original": COLORS["original"],
                  "GReaT": COLORS["great"],
                  "ChatGPT": COLORS["chatgpt"],
                  "Gemini": COLORS["gemini"]}

    fig, axes = plt.subplots(nrows_fig, ncols_fig,
                             figsize=(5*ncols_fig, 4*nrows_fig))
    axes = axes.flatten()

    for i, col in enumerate(all_cols):
        ax = axes[i]
        if col in num_cols:
            all_vals = pd.concat([df[col] for df in all_src.values()])
            lo, hi   = all_vals.min(), all_vals.max()
            bins = np.linspace(lo, hi, 20)
            for src_name, df in all_src.items():
                ax.hist(df[col], bins=bins, alpha=0.55, density=True,
                        color=src_colors[src_name], label=src_name)
        else:
            cats = sorted(train_real_raw[col].unique().tolist())
            x = np.arange(len(cats))
            w = 0.8 / len(all_src)
            for j, (src_name, df) in enumerate(all_src.items()):
                vc = df[col].value_counts(normalize=True)
                vals = [vc.get(c, 0) for c in cats]
                ax.bar(x + (j - len(all_src)/2 + 0.5)*w, vals, w,
                       color=src_colors[src_name], alpha=0.80,
                       label=src_name)
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in cats],
                               rotation=20, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 7: Column-wise Distributions — Real vs All Synthetic",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure7_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 9. Figure 8 — Radar chart (ML utility summary)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure8(store):
    """Radar chart: for each downstream model, Acc / AUC / F1 across methods."""
    methods  = METHOD_NAMES
    metrics  = ["acc", "auc", "f1"]
    m_labels = ["Accuracy", "ROCAUC", "F1"]
    clf_names = ["LR", "DT", "RF"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    # Baseline (original) for normalisation
    orig_vals = {m: {k: np.mean(store[m]["original"][k])
                     for k in metrics}
                 for m in clf_names}

    for ax, clf in zip(axes, clf_names):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), m_labels, fontsize=10)
        ax.set_ylim(50, 100)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.set_yticklabels(["60", "70", "80", "90", "100"], fontsize=7)

        # Original baseline (dashed)
        orig = [orig_vals[clf][k] for k in metrics] + \
               [orig_vals[clf][metrics[0]]]
        ax.plot(angles, orig, "k--", lw=1.5, label="Original")
        ax.fill(angles, orig, alpha=0.05, color="black")

        for name in methods:
            vals = [np.mean(store[clf][name][k]) for k in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, lw=2, color=COLORS[name],
                    label=DISPLAY[name])
            ax.fill(angles, vals, alpha=0.10, color=COLORS[name])

        ax.set_title(f"Classifier: {clf}", fontsize=11, pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    fig.suptitle("Figure 8: ML Utility Radar — GReaT vs ChatGPT vs Gemini "
                 "(Train on Synthetic, Test on Real)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure8_radar.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 10. Save CSVs
# ══════════════════════════════════════════════════════════════════════════════

def save_csvs(store, disc):
    rows = []
    for m in ["LR", "DT", "RF"]:
        for split in ["original"] + METHOD_NAMES:
            for metric in ["acc", "f1", "auc"]:
                vals = np.array(store[m][split][metric])
                rows.append({"classifier": m, "method": split,
                              "metric": metric,
                              "mean": round(vals.mean(), 4),
                              "std":  round(vals.std(), 4)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "tables.csv", index=False)

    disc_rows = [{"method": n,
                  "disc_acc_mean": round(np.mean(v), 4),
                  "disc_acc_std":  round(np.std(v),  4)}
                 for n, v in disc.items()]
    pd.DataFrame(disc_rows).to_csv(OUT_DIR / "discriminator.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Statistical fidelity summary
# ══════════════════════════════════════════════════════════════════════════════

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
            tv_vals.append(sum(abs(r.get(c,0)-s.get(c,0)) for c in cats)/2)
        print(f"{DISPLAY[name]:<10}  {np.mean(ks_vals):14.4f}  "
              f"{np.mean(ws_vals):16.4f}  {np.mean(tv_vals):15.4f}")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data ...")
    train_real, test_real, raws, feature_cols, cat_cols, encoders = load_all()

    # Keep raw (un-encoded) copies for distribution plots
    train_raw = pd.read_csv(TRAIN_PATH)
    raws_raw  = {"great":   pd.read_csv(GREAT_PATH).dropna().reset_index(drop=True),
                 "chatgpt": pd.read_csv(CHATGPT_PATH),
                 "gemini":  pd.read_csv(GEMINI_PATH)}

    # ── Overview ────────────────────────────────────────────────────────────────
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

    # ── Statistical fidelity ────────────────────────────────────────────────────
    print_fidelity(train_real, raws, feature_cols, cat_cols)

    # ── ML utility ─────────────────────────────────────────────────────────────
    print(f"\nRunning ML utility (5 trials x 3 models) ...")
    store = compute_ml(train_real, test_real, raws, feature_cols)
    print_table1(store)
    print_table4(store)

    # ── Discriminator ───────────────────────────────────────────────────────────
    print(f"\nRunning discriminator (5 trials x 3 methods) ...")
    disc = compute_discriminator(train_real, raws, feature_cols)
    print_table2(disc)

    # ── DCR ─────────────────────────────────────────────────────────────────────
    print(f"\nComputing DCR ...")
    dcr_test, dcr_syn = compute_dcr(train_real, test_real, raws, feature_cols)
    print(f"  Test   -> Train : mean={dcr_test.mean():.4f}")
    for n, d in dcr_syn.items():
        print(f"  {DISPLAY[n]:<10}-> Train : mean={d.mean():.4f}  "
              f"ratio={d.mean()/dcr_test.mean():.3f}")

    # ── Plots ───────────────────────────────────────────────────────────────────
    print(f"\nGenerating figures ...")
    plot_figure5(dcr_test, dcr_syn)
    plot_figure6(train_real, raws)
    plot_figure7(train_raw, raws_raw, feature_cols, cat_cols, encoders)
    plot_figure8(store)

    # ── Save ────────────────────────────────────────────────────────────────────
    save_csvs(store, disc)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
