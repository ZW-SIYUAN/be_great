"""
evaluate_spatial.py
───────────────────
Spatial Drift Evaluation for the Covertype dataset, structured to match 
the temporal drift evaluation (evaluate_drift.py) used for the electricity dataset.

  Temporal Drift (Electricity)        Spatial Drift (Covertype)
  ─────────────────────────        ──────────────────────────────
  1. Rolling P(UP)                 1. Cover_Type Dist per Area (Bar)
  2. Feature Rolling Means         2. Feature Means per Area (Bar)
  3. Sliding Window KS             3. Inter-area KS Matrix (Heatmap, 4x4)
  4. Segmented Accuracy            4. Cross-area Accuracy Matrix (Heatmap, 4x4)
  5. Quartile Dist Comparison      5. Key Feature Dist per Area (Violin)
  6. Quantitative Summary          6. Quantitative Summary

Core Question: Does data synthesized by GReaT retain the spatial heterogeneity 
(differences in feature/label distribution and generalization difficulty) 
across the 4 Wilderness Areas?

Prerequisite: Run train_and_sample.py first to generate covertype_synthetic.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(HERE, filename)

warnings.filterwarnings("ignore", category=UserWarning)

# --- Data Loading ----------------------------------------------------
real  = pd.read_csv(data_path("covertype_train.csv")).reset_index(drop=True)
synth = pd.read_csv(data_path("covertype_synthetic.csv")).reset_index(drop=True)

NUM_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
SPATIAL_COL = "Wilderness_Area"   # Spatial partition axis
LABEL_COL   = "Cover_Type"        # Prediction target
AREAS       = ["1", "2", "3", "4"]
AREA_NAMES  = {
    "1": "Rawah", "2": "Neota",
    "3": "Comanche\nPeak", "4": "Cache la\nPoudre",
}

# Ensure numerical columns are float
for col in NUM_COLS:
    real[col]  = pd.to_numeric(real[col],  errors="coerce")
    synth[col] = pd.to_numeric(synth[col], errors="coerce")

# Ensure spatial/label columns are string and filter invalid hallucinations
synth[SPATIAL_COL] = synth[SPATIAL_COL].astype(str).str.strip()
synth[LABEL_COL]   = synth[LABEL_COL].astype(str).str.strip()
real[SPATIAL_COL]  = real[SPATIAL_COL].astype(str).str.strip()
real[LABEL_COL]    = real[LABEL_COL].astype(str).str.strip()

valid_areas  = set(AREAS)
valid_labels = set(real[LABEL_COL].unique())
synth = synth[synth[SPATIAL_COL].isin(valid_areas) &
              synth[LABEL_COL].isin(valid_labels)].reset_index(drop=True)

# --- Global Styles ----------------------------------------------------
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
REAL_COLOR, SYNTH_COLOR = "#1f77b4", "#ff7f0e"
AREA_COLORS = ["#1a9850", "#91cf60", "#fc8d59", "#d73027"]

print(f"Real : {real.shape}  Samples per Area: { {a: int((real[SPATIAL_COL]==a).sum()) for a in AREAS} }")
print(f"Synth: {synth.shape}  Samples per Area: { {a: int((synth[SPATIAL_COL]==a).sum()) for a in AREAS} }")

# ================================================================
# Helper Functions
# ================================================================

def get_area(df, area):
    return df[df[SPATIAL_COL] == area]

def cover_type_dist(df, area):
    """Returns the ratio Series of Cover_Type for a specific area."""
    sub = get_area(df, area)
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return sub[LABEL_COL].value_counts(normalize=True).sort_index()

def cross_area_accuracy(df, feature_cols, label_col):
    """
    4x4 Cross-area accuracy matrix.
    entry(i,j) = Accuracy of RandomForest trained on Area i and tested on Area j.
    """
    le = LabelEncoder()
    all_labels = df[label_col].dropna().unique()
    le.fit(all_labels)

    matrix = np.full((4, 4), np.nan)
    for i, a_train in enumerate(AREAS):
        tr_df = get_area(df, a_train)
        if len(tr_df) < 20: continue
        for j, a_test in enumerate(AREAS):
            if i == j:
                split = int(len(tr_df) * 0.8)
                X_tr = tr_df.iloc[:split][feature_cols].fillna(0).values
                y_tr = le.transform(tr_df.iloc[:split][label_col])
                X_te = tr_df.iloc[split:][feature_cols].fillna(0).values
                y_te = le.transform(tr_df.iloc[split:][label_col])
            else:
                te_df = get_area(df, a_test)
                if len(te_df) < 10: continue
                X_tr = tr_df[feature_cols].fillna(0).values
                y_tr = le.transform(tr_df[label_col])
                X_te = te_df[feature_cols].fillna(0).values
                y_te = le.transform(te_df[label_col])

            if len(np.unique(y_tr)) < 2 or len(X_te) == 0: continue
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_tr, y_tr)
            matrix[i, j] = accuracy_score(y_te, clf.predict(X_te))
    return matrix

def area_ks_matrix(df, col):
    """4x4 Inter-area KS distance matrix."""
    matrix = np.zeros((4, 4))
    for i, ai in enumerate(AREAS):
        for j, aj in enumerate(AREAS):
            vi = get_area(df, ai)[col].dropna().values
            vj = get_area(df, aj)[col].dropna().values
            if len(vi) < 5 or len(vj) < 5:
                matrix[i, j] = np.nan
            elif i == j:
                matrix[i, j] = 0.0
            else:
                ks, _ = stats.ks_2samp(vi, vj)
                matrix[i, j] = ks
    return matrix

# ================================================================
# Fig 1: Cover_Type Dist per Area (Benchmark: Rolling P(UP))
# ================================================================
all_types = sorted(set(real[LABEL_COL].unique()) | set(synth[LABEL_COL].unique()),
                   key=lambda x: int(x) if x.isdigit() else x)
type_labels = [f"Type {t}" for t in all_types]
x = np.arange(len(all_types))
bar_w = 0.18

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
for ax, area, c in zip(axes, AREAS, AREA_COLORS):
    r_dist = cover_type_dist(real,  area).reindex(all_types, fill_value=0)
    s_dist = cover_type_dist(synth, area).reindex(all_types, fill_value=0)
    ax.bar(x - bar_w / 2, r_dist.values, width=bar_w,
           color=REAL_COLOR,  label="Real",      alpha=0.85)
    ax.bar(x + bar_w / 2, s_dist.values, width=bar_w,
           color=SYNTH_COLOR, label="Synthetic",  alpha=0.85, hatch="//")
    ax.set_title(f"Area {area}\n{AREA_NAMES[area]}", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=45, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=0.3)
    if ax == axes[0]: ax.legend(fontsize=8)

fig.suptitle("Spatial Drift: Cover_Type Distribution per Wilderness Area\n"
             "Real (solid) vs Synthetic (hatched) — Benchmarked against electricity P(UP) distribution",
             fontweight="bold")
plt.tight_layout()
plt.savefig(data_path("spatial_1_covertype_dist.png"))
plt.show()
print("[1] Saved spatial_1_covertype_dist.png")

# ================================================================
# Fig 2: Per-Area Feature Means (Benchmark: Rolling Feature Means)
# ================================================================
KEY_NUM = ["Elevation", "Slope", "Horizontal_Distance_To_Hydrology",
           "Hillshade_9am", "Horizontal_Distance_To_Roadways",
           "Horizontal_Distance_To_Fire_Points"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
x_pos = np.arange(4)
for ax, col in zip(axes, KEY_NUM):
    r_means = [get_area(real,  a)[col].mean() for a in AREAS]
    s_means = [get_area(synth, a)[col].mean() for a in AREAS]
    ax.bar(x_pos - 0.2, r_means, 0.4, color=REAL_COLOR,  label="Real",     alpha=0.85)
    ax.bar(x_pos + 0.2, s_means, 0.4, color=SYNTH_COLOR, label="Synthetic", alpha=0.85, hatch="//")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Area {a}" for a in AREAS], fontsize=8)
    ax.set_title(col, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)

fig.suptitle("Spatial Drift: Per-Area Feature Means — Real vs Synthetic\n"
             "Benchmarked against electricity rolling feature mean plots",
             fontweight="bold")
plt.tight_layout()
plt.savefig(data_path("spatial_2_feature_means.png"))
plt.show()
print("[2] Saved spatial_2_feature_means.png")

# ================================================================
# Fig 3: Inter-area KS Matrix (Benchmark: Rolling Window KS Curve)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
area_tick_labels = [f"Area {a}\n{AREA_NAMES[a]}" for a in AREAS]

for ax, df, title in zip(axes, [real, synth],
                          ["Real Data — KS(Elevation) between Areas",
                           "Synthetic Data — KS(Elevation) between Areas"]):
    mat = area_ks_matrix(df, "Elevation")
    im  = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(4)); ax.set_xticklabels(area_tick_labels, fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels(area_tick_labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="white" if val > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("Between-Area KS Distance Matrix (Elevation)\n"
             "Benchmarked against sliding KS curve | Pattern similarity → Better spatial structure retention",
             fontweight="bold")
plt.tight_layout()
plt.savefig(data_path("spatial_3_ks_matrix.png"))
plt.show()
print("[3] Saved spatial_3_ks_matrix.png")

# ================================================================
# Fig 4: Cross-Area Accuracy Matrix (Benchmark: Segmented Accuracy)
# ================================================================
print("\nComputing Cross-Area Accuracy Matrix (Real)...")
acc_real  = cross_area_accuracy(real,  NUM_COLS, LABEL_COL)
print("Computing Cross-Area Accuracy Matrix (Synth)...")
acc_synth = cross_area_accuracy(synth, NUM_COLS, LABEL_COL)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mat, title in zip(axes, [acc_real, acc_synth],
                            ["Real Data — Cross-Area Accuracy",
                             "Synthetic Data — Cross-Area Accuracy"]):
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"Test\nArea {a}" for a in AREAS], fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"Train\nArea {a}" for a in AREAS], fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color="white" if val < 0.4 or val > 0.85 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("Cross-Area Classification Accuracy Matrix\n"
             "Low off-diagonal accuracy = Strong Spatial Drift | Higher pattern similarity = Better GReaT",
             fontweight="bold")
plt.tight_layout()
plt.savefig(data_path("spatial_4_crossarea_accuracy.png"))
plt.show()
print("[4] Saved spatial_4_crossarea_accuracy.png")

# ================================================================
# Fig 5: Key Feature Dist per Area (Violin, Benchmark: Quartile Dist)
# ================================================================
VIOLIN_COLS = ["Elevation", "Slope", "Horizontal_Distance_To_Hydrology"]
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

for ax, col in zip(axes, VIOLIN_COLS):
    real_data_by_area  = [get_area(real,  a)[col].dropna().values for a in AREAS]
    synth_data_by_area = [get_area(synth, a)[col].dropna().values for a in AREAS]

    positions_r = np.array([1, 4, 7, 10], dtype=float)
    positions_s = positions_r + 1.2

    vp_r = ax.violinplot(real_data_by_area,  positions=positions_r,
                         widths=1.0, showmedians=True, showextrema=False)
    vp_s = ax.violinplot(synth_data_by_area, positions=positions_s,
                         widths=1.0, showmedians=True, showextrema=False)

    for body in vp_r["bodies"]:
        body.set_facecolor(REAL_COLOR); body.set_alpha(0.6)
    vp_r["cmedians"].set_color(REAL_COLOR)
    for body in vp_s["bodies"]:
        body.set_facecolor(SYNTH_COLOR); body.set_alpha(0.6)
    vp_s["cmedians"].set_color(SYNTH_COLOR)

    ax.set_xticks(positions_r + 0.6)
    ax.set_xticklabels([f"Area {a}\n{AREA_NAMES[a]}" for a in AREAS], fontsize=7)
    ax.set_title(col, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=REAL_COLOR,  alpha=0.7, label="Real"),
              Patch(facecolor=SYNTH_COLOR, alpha=0.7, label="Synthetic")]
axes[0].legend(handles=legend_els, fontsize=8)

fig.suptitle("Spatial Distribution by Wilderness Area — Violin Plots\n"
             "Real (Blue) vs Synthetic (Orange) | Differences across areas indicate Spatial Drift",
             fontweight="bold")
plt.tight_layout()
plt.savefig(data_path("spatial_5_violin_dist.png"))
plt.show()
print("[5] Saved spatial_5_violin_dist.png")

# ================================================================
# 6. Quantitative Summary
# ================================================================
SEP = "=" * 65
print(f"\n{SEP}")
print("Quantitative Summary".center(65))
print(f"{SEP}")

# [A] Within-Area KS: Real vs Synth
print("\n[A] Within-Area KS Test (Real_area vs Synth_area) - Lower is more accurate")
print(f"  {'Feature':40s}", end="")
for a in AREAS: print(f"  Area{a:1s}", end="")
print()
for col in NUM_COLS:
    print(f"  {col:40s}", end="")
    for area in AREAS:
        r_vals = get_area(real,  area)[col].dropna().values
        s_vals = get_area(synth, area)[col].dropna().values
        if len(r_vals) < 5 or len(s_vals) < 5: print("   N/A ", end="")
        else:
            ks, _ = stats.ks_2samp(r_vals, s_vals)
            print(f"  {ks:.3f}", end="")
    print()

# [B] Inter-area KS Matrix Correlation
print("\n[B] Correlation of Inter-area KS Matrices (Real Matrix vs Synth Matrix)")
print("    r ≈ 1 means GReaT preserved the relative spatial structure")
for col in ["Elevation", "Slope", "Horizontal_Distance_To_Hydrology",
            "Hillshade_9am", "Horizontal_Distance_To_Fire_Points"]:
    mat_r, mat_s = area_ks_matrix(real, col), area_ks_matrix(synth, col)
    idx = np.triu_indices(4, k=1)
    r_flat, s_flat = mat_r[idx], mat_s[idx]
    mask = ~(np.isnan(r_flat) | np.isnan(s_flat))
    if mask.sum() < 2: print(f"  {col:40s}  r=N/A"); continue
    try:
        rho, p = stats.pearsonr(r_flat[mask], s_flat[mask])
        print(f"  {col:40s}  r={rho:+.4f}  p={p:.2e}")
    except: print(f"  {col:40s}  r=N/A")

# [C] Cross-area Accuracy Summary
print("\n[C] Cross-Area Classification Accuracy Summary")
off_diag_idx = [(i, j) for i in range(4) for j in range(4) if i != j]
r_off = [acc_real[i, j]  for i, j in off_diag_idx if not np.isnan(acc_real[i, j])]
s_off = [acc_synth[i, j] for i, j in off_diag_idx if not np.isnan(acc_synth[i, j])]
r_diag = [acc_real[i, i]  for i in range(4) if not np.isnan(acc_real[i, i])]
s_diag = [acc_synth[i, i] for i in range(4) if not np.isnan(acc_synth[i, i])]

print(f"  Real  Diagonal (In-area) Mean: {np.mean(r_diag):.4f}")
print(f"  Synth Diagonal (In-area) Mean: {np.mean(s_diag):.4f}")
print(f"  Real  Off-diagonal (Cross-area) Mean: {np.mean(r_off):.4f} (Gap: {np.mean(r_diag)-np.mean(r_off):.4f})")
print(f"  Synth Off-diagonal (Cross-area) Mean: {np.mean(s_off):.4f} (Gap: {np.mean(s_diag)-np.mean(s_off):.4f})")

# [D] Cover_Type TVD
print("\n[D] Global Cover_Type TVD (Total Variation Distance, lower is better)")
for area in AREAS:
    r_d = cover_type_dist(real,  area).reindex(all_types, fill_value=0)
    s_d = cover_type_dist(synth, area).reindex(all_types, fill_value=0)
    tvd = float(np.abs(r_d.values - s_d.values).sum() / 2)
    print(f"  Area {area} ({AREA_NAMES[area]:14s})  TVD = {tvd:.4f}")

print(f"\n{SEP}")
print("Spatial Drift Evaluation Complete. 5 figures saved to current directory.")