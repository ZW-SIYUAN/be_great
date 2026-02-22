"""
Reproduce paper Tables 1, 2, 4 and Figure 5 for the Sick (Thyroid) dataset.
Evaluates GReaT-generated synthetic data (tabularisai/Qwen3-0.3B-distil, LoRA).

Fixed splits (no randomness in data partitioning):
  sick_train.csv    — 80 % real data  (used to fine-tune GReaT)
  sick_test.csv     — 20 % real data  (held-out evaluation set)
  sick_synthetic.csv — GReaT synthetic data (same size as train set)

Table 1  — ML Utility: Accuracy (%)        | LR / DT / RF | Original vs GReaT
Table 2  — Discriminator Accuracy (%)      | RF           | GReaT (lower=better)
Table 4  — ML Utility: ROCAUC + F1 (%)    | LR / DT / RF | GReaT
Figure 5 — DCR distributions               | GReaT vs Train, baseline = Test vs Train

5 trials per experiment: fixed data splits, varying model random seed only.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist

from sklearn.base import clone as sklearn_clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

TARGET        = "Class"
VALID_CLASSES = {"negative", "sick"}   # filter out malformed GReaT rows
SEEDS         = [42, 1, 2, 3, 4]      # 5 trials (seed for model init only)
DISC_TEST_SIZE = 0.30                  # discriminator hold-out fraction

# TBG is entirely missing in both real and synthetic data — drop it.
DROP_COLS = ["TBG"]


# ── 1. Data loading & encoding ────────────────────────────────────────────────

def _clean(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop all-null columns, filter rows with unrecognised target labels,
    and report any dropped rows."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    before = len(df)
    df = df[df[TARGET].isin(VALID_CLASSES)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[info] {label}: dropped {dropped} malformed row(s) "
              f"({before} -> {len(df)})")
    return df


def load_all():
    train_real = pd.read_csv(HERE / "sick_train.csv")
    test_real  = pd.read_csv(HERE / "sick_test.csv")
    syn_great  = pd.read_csv(HERE / "sick_synthetic.csv")

    train_real = _clean(train_real, "train_real")
    test_real  = _clean(test_real,  "test_real")
    syn_great  = _clean(syn_great,  "syn_great")

    all_dfs = [train_real, test_real, syn_great]

    # ── Encode categoricals (using union of all datasets) ──────────────────────
    cat_cols = [c for c in train_real.columns
                if train_real[c].dtype == "object" and c != TARGET]
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([d[col] for d in all_dfs]).astype(str)
        le.fit(combined)
        for d in all_dfs:
            d[col] = le.transform(d[col].astype(str))

    # Encode target
    le_target = LabelEncoder()
    combined_target = pd.concat([d[TARGET] for d in all_dfs]).astype(str)
    le_target.fit(combined_target)
    for d in all_dfs:
        d[TARGET] = le_target.transform(d[TARGET].astype(str))

    # ── Impute missing numerics with per-column median (train statistics) ──────
    feature_cols = [c for c in train_real.columns if c != TARGET]
    num_cols = [c for c in feature_cols if train_real[c].dtype != "object"]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_real[num_cols])
    for d in all_dfs:
        d[num_cols] = imputer.transform(d[num_cols])

    return train_real, test_real, syn_great, feature_cols


# ── 2. Classifiers ────────────────────────────────────────────────────────────

def get_classifiers(seed):
    return {
        "LR": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed,
                                     n_jobs=-1),
    }


# ── 3. Single ML trial ────────────────────────────────────────────────────────

def run_ml_trial(train_real, test_real, syn_dict, feature_cols, seed):
    """
    Returns {model: {split: {acc, f1, auc}}}
    split keys: 'original', and one key per entry in syn_dict.
    """
    X_te = test_real[feature_cols].values
    y_te = test_real[TARGET].values

    splits = {"original": (train_real[feature_cols].values,
                           train_real[TARGET].values)}
    splits.update({name: (df[feature_cols].values, df[TARGET].values)
                   for name, df in syn_dict.items()})

    results = {}
    for model_name, clf in get_classifiers(seed).items():
        row = {}
        for split_name, (Xtr, ytr) in splits.items():
            c = sklearn_clone(clf)
            c.fit(Xtr, ytr)
            yp  = c.predict(X_te)
            ypr = c.predict_proba(X_te)[:, 1]
            row[split_name] = {
                "acc": accuracy_score(y_te, yp) * 100,
                "f1":  f1_score(y_te, yp, average="weighted",
                                zero_division=0) * 100,
                "auc": roc_auc_score(y_te, ypr) * 100,
            }
        results[model_name] = row
    return results


# ── 4. Aggregate ML over 5 trials ─────────────────────────────────────────────

def compute_ml_tables(train_real, test_real, syn_dict, feature_cols):
    models     = ["LR", "DT", "RF"]
    split_keys = ["original"] + list(syn_dict.keys())
    metrics    = ["acc", "f1", "auc"]

    store = {m: {s: {k: [] for k in metrics} for s in split_keys}
             for m in models}

    for seed in SEEDS:
        trial = run_ml_trial(train_real, test_real, syn_dict, feature_cols, seed)
        for m in models:
            for s in split_keys:
                for k in metrics:
                    store[m][s][k].append(trial[m][s][k])
    return store


# ── 5. Discriminator (Table 2) ────────────────────────────────────────────────

def run_discriminator_trial(train_real, syn_df, feature_cols, seed):
    """
    RF trained to distinguish real train (1) vs synthetic (0).
    Returns hold-out accuracy (%).
    """
    X = np.vstack([train_real[feature_cols].values,
                   syn_df[feature_cols].values])
    y = np.array([1] * len(train_real) + [0] * len(syn_df))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=DISC_TEST_SIZE, stratify=y, random_state=seed
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)) * 100


def compute_discriminator(train_real, syn_dict, feature_cols):
    return {name: [run_discriminator_trial(train_real, df, feature_cols, s)
                   for s in SEEDS]
            for name, df in syn_dict.items()}


# ── 6. DCR (Figure 5) ────────────────────────────────────────────────────────

def compute_and_plot_dcr(train_real, test_real, syn_dict, feature_cols):
    X_train = train_real[feature_cols].values.astype(float)
    X_test  = test_real[feature_cols].values.astype(float)

    # Normalise with train statistics
    col_min = X_train.min(0)
    col_max = X_train.max(0)
    rng = np.where(col_max > col_min, col_max - col_min, 1.0)

    Xtr_n  = (X_train - col_min) / rng
    Xte_n  = (X_test  - col_min) / rng

    dcr_test = cdist(Xte_n, Xtr_n, metric="euclidean").min(axis=1)

    colors = {"great": "darkorange"}
    all_vals = [dcr_test]

    syn_dcrs = {}
    for name, df in syn_dict.items():
        Xs_n = (df[feature_cols].values.astype(float) - col_min) / rng
        dcr  = cdist(Xs_n, Xtr_n, metric="euclidean").min(axis=1)
        syn_dcrs[name] = dcr
        all_vals.append(dcr)

    hi = max(v.max() for v in all_vals) * 1.05
    bins = np.linspace(0, hi, 55)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dcr_test, bins=bins, alpha=0.50, color="steelblue", density=True,
            label=f"Original Test Set  (mean={dcr_test.mean():.3f})")
    for name, dcr in syn_dcrs.items():
        ax.hist(dcr, bins=bins, alpha=0.50, color=colors.get(name, "gray"),
                density=True,
                label=f"{name.capitalize()} Synthetic  (mean={dcr.mean():.3f})")

    ax.set_xlabel("Distance to Closest Record (L2, normalised)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Figure 5: DCR Distributions — Sick (Thyroid) Dataset\n"
                 "(Synthetic Data vs. Original Train Set)", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure5_dcr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nDCR summary:")
    print(f"  Test   -> Train : mean={dcr_test.mean():.4f}  "
          f"median={np.median(dcr_test):.4f}")
    for name, dcr in syn_dcrs.items():
        ratio = dcr.mean() / (dcr_test.mean() + 1e-10)
        print(f"  {name:<10}-> Train : mean={dcr.mean():.4f}  "
              f"median={np.median(dcr):.4f}  ratio={ratio:.3f}")
    return syn_dcrs, dcr_test


# ── 7. Print tables ───────────────────────────────────────────────────────────

SEP = "=" * 76

def _ms(vals):
    a = np.array(vals)
    return f"{a.mean():6.2f}+-{a.std():.2f}"


def print_table1(store, syn_names):
    print(f"\n{SEP}")
    print("TABLE 1  ML Utility — Accuracy (%)  |  Dataset: Sick (Thyroid)")
    header = f"{'Model':<6}  {'Original':>16}" + \
             "".join(f"  {n.capitalize():>16}" for n in syn_names) + \
             "".join(f"  {'d_'+n[:2]:>8}" for n in syn_names)
    print(header)
    print("-" * len(header))
    for m in ["LR", "DT", "RF"]:
        orig = np.array(store[m]["original"]["acc"])
        line = f"{m:<6}  {_ms(store[m]['original']['acc'])}"
        for name in syn_names:
            line += f"  {_ms(store[m][name]['acc'])}"
        for name in syn_names:
            delta = np.mean(store[m][name]["acc"]) - orig.mean()
            line += f"  {delta:+7.2f}"
        print(line)
    print(SEP)


def print_table2(disc):
    print(f"\n{SEP}")
    print("TABLE 2  Discriminator Accuracy (%)  |  lower=better, 50%=indistinguishable")
    print(SEP)
    for name, vals in disc.items():
        a = np.array(vals)
        print(f"  {name.capitalize():<12} {a.mean():.2f}+-{a.std():.2f}  "
              f"trials={[round(v,2) for v in vals]}")
    print(SEP)


def print_table4(store, syn_names):
    print(f"\n{SEP}")
    print("TABLE 4  ML Utility — ROCAUC (%) and F1 (%)  |  Dataset: Sick (Thyroid)")
    header = f"{'Model':<6}" + \
             "".join(f"  {n.capitalize()+' AUC':>16}  {n.capitalize()+' F1':>16}"
                     for n in syn_names)
    print(header)
    print("-" * len(header))
    for m in ["LR", "DT", "RF"]:
        line = f"{m:<6}"
        for name in syn_names:
            line += f"  {_ms(store[m][name]['auc'])}  {_ms(store[m][name]['f1'])}"
        print(line)
    print(SEP)


# ── 8. Save CSV ───────────────────────────────────────────────────────────────

def save_results(store, disc, syn_names):
    rows = []
    for m in ["LR", "DT", "RF"]:
        row = {"Model": m}
        for split in ["original"] + syn_names:
            for metric in ["acc", "f1", "auc"]:
                vals = np.array(store[m][split][metric])
                row[f"{split}_{metric}_mean"] = round(vals.mean(), 4)
                row[f"{split}_{metric}_std"]  = round(vals.std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "table1_table4.csv", index=False)

    disc_rows = []
    for name, vals in disc.items():
        a = np.array(vals)
        disc_rows.append({"method": name,
                          "disc_acc_mean": round(a.mean(), 4),
                          "disc_acc_std":  round(a.std(), 4),
                          "trials": vals})
    pd.DataFrame(disc_rows).to_csv(OUT_DIR / "table2.csv", index=False)


# ── 9. Dataset overview ───────────────────────────────────────────────────────

def print_overview(train_real, test_real, syn_dict):
    print(f"\n{SEP}")
    print("DATASET OVERVIEW")
    print(SEP)
    def tgt(df): return df[TARGET].value_counts().sort_index().to_dict()
    print(f"  train_real  : {len(train_real):5d} rows  Target={tgt(train_real)}")
    print(f"  test_real   : {len(test_real):5d} rows  Target={tgt(test_real)}")
    for name, df in syn_dict.items():
        print(f"  {name:<12}: {len(df):5d} rows  Target={tgt(df)}")
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading and encoding data ...")
    train_real, test_real, syn_great, feature_cols = load_all()

    syn_dict  = {"great": syn_great}
    syn_names = list(syn_dict.keys())

    print_overview(train_real, test_real, syn_dict)

    # ── Table 1 & 4 ────────────────────────────────────────────────────────────
    print(f"\nRunning ML utility (5 trials x 3 models x 3 splits) ...")
    store = compute_ml_tables(train_real, test_real, syn_dict, feature_cols)
    print_table1(store, syn_names)
    print_table4(store, syn_names)

    # ── Table 2 ────────────────────────────────────────────────────────────────
    print(f"\nRunning discriminator (5 trials x 2 methods) ...")
    disc = compute_discriminator(train_real, syn_dict, feature_cols)
    print_table2(disc)

    # ── Figure 5 ───────────────────────────────────────────────────────────────
    print(f"\nComputing DCR distributions ...")
    compute_and_plot_dcr(train_real, test_real, syn_dict, feature_cols)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_results(store, disc, syn_names)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
