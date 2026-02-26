"""
explore_data.py
---------------
Visualize the distribution of covertype_train.csv (preprocessed data).
Run this before train_and_sample.py to understand data structure
and spatial differences between Wilderness Areas (source of spatial concept drift).

Prerequisite: run prepare_data.py first to generate covertype_train.csv

Output files (5 total):
  explore_1_overview.png         Target variable & categorical columns overview
  explore_2_num_dist.png         Histograms of 10 numerical features
  explore_3_spatial_label.png    Cover_Type distribution per area (spatial drift)
  explore_4_spatial_features.png Box plots of numerical features per area
  explore_5_corr_scatter.png     Correlation heatmap + scatter plot

Run:
  cd "concept drift exeperiment/Covertype"
  python explore_data.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# -- Path: works regardless of which directory the script is run from -----------
HERE = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(HERE, filename)

# -- Load data -----------------------------------------------------------------
df = pd.read_csv(data_path("covertype_train.csv"),
                 dtype={"Wilderness_Area": str, "Soil_Type": str, "Cover_Type": str})

NUM_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
SPATIAL_COL = "Wilderness_Area"
SOIL_COL    = "Soil_Type"
LABEL_COL   = "Cover_Type"
AREAS       = ["1", "2", "3", "4"]
AREA_NAMES  = {"1": "Rawah", "2": "Neota", "3": "Comanche Peak", "4": "Cache la Poudre"}
AREA_COLORS = ["#2166ac", "#4dac26", "#d01c8b", "#f1a340"]
COVER_COLORS= ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f"]

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

print(f"Data shape: {df.shape}")
print(f"\nCover_Type distribution:\n{df[LABEL_COL].value_counts().sort_index()}")
print(f"\nWilderness_Area distribution:\n{df[SPATIAL_COL].value_counts().sort_index()}")

# ================================================================
# Figure 1: Target variable & categorical columns overview
# ================================================================
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Covertype Preprocessed Data Overview", fontsize=13, fontweight="bold", y=1.01)

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# 1a. Cover_Type overall distribution
ax1 = fig.add_subplot(gs[0, 0])
ct_counts = df[LABEL_COL].value_counts().sort_index()
bars = ax1.bar(ct_counts.index, ct_counts.values, color=COVER_COLORS[:len(ct_counts)], edgecolor="white")
ax1.set_title("Cover_Type Overall Distribution (Target)")
ax1.set_xlabel("Cover_Type")
ax1.set_ylabel("Count")
for bar, v in zip(bars, ct_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(v), ha="center", va="bottom", fontsize=8)
ax1.grid(axis="y", alpha=0.3)

# Cover_Type legend text
type_desc = {
    "1": "Spruce/Fir", "2": "Lodgepole Pine", "3": "Ponderosa Pine",
    "4": "Cottonwood/Willow", "5": "Aspen", "6": "Douglas-fir", "7": "Krummholz"
}
desc_text = "\n".join([f"Type {k}: {v}" for k, v in type_desc.items()])
ax1.text(1.02, 0.5, desc_text, transform=ax1.transAxes,
         fontsize=7, va="center",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# 1b. Wilderness_Area overall distribution
ax2 = fig.add_subplot(gs[0, 1])
wa_counts = df[SPATIAL_COL].value_counts().sort_index()
bars2 = ax2.bar([f"Area {a}\n{AREA_NAMES[a]}" for a in wa_counts.index],
                wa_counts.values, color=AREA_COLORS, edgecolor="white")
ax2.set_title("Wilderness_Area Distribution (Spatial Axis)")
ax2.set_ylabel("Count")
for bar, v in zip(bars2, wa_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(v), ha="center", va="bottom", fontsize=8)
ax2.grid(axis="y", alpha=0.3)

# 1c. Soil_Type distribution
ax3 = fig.add_subplot(gs[0, 2])
st_counts = df[SOIL_COL].value_counts().sort_index(key=lambda x: x.astype(int))
ax3.bar(range(len(st_counts)), st_counts.values, color="#888888", edgecolor="white", width=0.8)
ax3.set_title("Soil_Type Distribution (40 types, merged from one-hot)")
ax3.set_xlabel("Soil_Type (1-40)")
ax3.set_ylabel("Count")
ax3.set_xticks(range(0, len(st_counts), 5))
ax3.set_xticklabels([list(st_counts.index)[i] for i in range(0, len(st_counts), 5)], fontsize=7)
ax3.grid(axis="y", alpha=0.3)

# 1d. Per-area Cover_Type pie charts
cover_types = sorted(df[LABEL_COL].unique(), key=int)
for idx, area in enumerate(AREAS):
    ax = fig.add_subplot(gs[1, idx if idx < 3 else 2])
    sub = df[df[SPATIAL_COL] == area]
    ct_a = sub[LABEL_COL].value_counts().reindex(cover_types, fill_value=0)
    wedges, texts, autotexts = ax.pie(
        ct_a.values, labels=None,
        colors=COVER_COLORS[:len(cover_types)],
        autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 0.5}
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.set_title(f"Area {area} - {AREA_NAMES[area]}\n(n={len(sub)})", fontsize=8)

legend_patches = [mpatches.Patch(color=COVER_COLORS[i], label=f"Type {ct}: {type_desc[ct]}")
                  for i, ct in enumerate(cover_types)]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=7, bbox_to_anchor=(0.5, -0.04))

plt.savefig(data_path("explore_1_overview.png"), bbox_inches="tight")
plt.show()
print("[1] Saved explore_1_overview.png")

# ================================================================
# Figure 2: Histogram + KDE for all 10 numerical features
# ================================================================
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for ax, col in zip(axes, NUM_COLS):
    vals = df[col].dropna()
    n_bins = min(50, int(np.sqrt(len(vals))))
    ax.hist(vals, bins=n_bins, color="#4292c6", edgecolor="white",
            alpha=0.85, density=True)
    # KDE curve
    kde_x = np.linspace(vals.min(), vals.max(), 300)
    kde_y = stats.gaussian_kde(vals)(kde_x)
    ax.plot(kde_x, kde_y, color="#08306b", lw=1.5)

    ax.set_title(col, fontsize=8)
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=7)

    ax.axvline(vals.mean(),   color="red",    lw=1.2, ls="--", label=f"Mean={vals.mean():.0f}")
    ax.axvline(vals.median(), color="orange", lw=1.2, ls=":",  label=f"Median={vals.median():.0f}")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(alpha=0.25)

fig.suptitle("Covertype Numerical Feature Distributions (Histogram + KDE)\n"
             "Red dashed = Mean  |  Orange dotted = Median",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(data_path("explore_2_num_dist.png"))
plt.show()
print("[2] Saved explore_2_num_dist.png")

# ================================================================
# Figure 3: Cover_Type distribution per Wilderness Area
#           (Core spatial drift visualization)
# ================================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
x = np.arange(len(cover_types))

for ax, area, c in zip(axes, AREAS, AREA_COLORS):
    sub = df[df[SPATIAL_COL] == area]
    ct_a = sub[LABEL_COL].value_counts().reindex(cover_types, fill_value=0)
    pct  = ct_a.values / ct_a.sum()
    bars = ax.bar(x, pct, color=[COVER_COLORS[i] for i in range(len(cover_types))],
                  edgecolor="white", width=0.7)
    ax.set_title(f"Area {area}\n{AREA_NAMES[area]}\n(n={len(sub)})", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{ct}" for ct in cover_types], fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    for bar, p in zip(bars, pct):
        if p > 0.03:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{p:.0%}", ha="center", va="bottom", fontsize=7)

# Global average as reference line
global_pct = df[LABEL_COL].value_counts().reindex(cover_types, fill_value=0)
global_pct = global_pct / global_pct.sum()
for ax in axes:
    for i, p in enumerate(global_pct.values):
        ax.hlines(p, i - 0.35, i + 0.35, colors="black", lw=1.2, ls="--", alpha=0.5)

fig.suptitle("Cover_Type Distribution per Wilderness Area - Spatial Drift Visualization\n"
             "Black dashed = global average  |  Bar height differences = source of spatial concept drift",
             fontsize=11, fontweight="bold")
legend_patches = [mpatches.Patch(color=COVER_COLORS[i], label=f"Type {ct}: {type_desc[ct]}")
                  for i, ct in enumerate(cover_types)]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=8, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
plt.savefig(data_path("explore_3_spatial_label.png"), bbox_inches="tight")
plt.show()
print("[3] Saved explore_3_spatial_label.png")

# ================================================================
# Figure 4: Box plots of numerical features per area
# ================================================================
KEY_COLS = ["Elevation", "Slope", "Horizontal_Distance_To_Hydrology",
            "Hillshade_9am", "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, col in zip(axes, KEY_COLS):
    data_by_area = [df[df[SPATIAL_COL] == a][col].dropna().values for a in AREAS]
    bp = ax.boxplot(data_by_area, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2),
                    flierprops=dict(marker=".", ms=2, alpha=0.3))
    for patch, c in zip(bp["boxes"], AREA_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticklabels([f"Area {a}\n{AREA_NAMES[a]}" for a in AREAS], fontsize=7)
    ax.set_title(col, fontsize=9)
    ax.set_ylabel("Value", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Annotate mean value per area
    for i, vals in enumerate(data_by_area):
        ax.text(i + 1, np.median(vals),
                f"{np.mean(vals):.0f}", ha="center", va="bottom",
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6))

fig.suptitle("Key Numerical Features by Wilderness Area (Box Plots)\n"
             "Box color = area  |  Black line = median  |  Text label = mean",
             fontsize=11, fontweight="bold")
legend_patches = [mpatches.Patch(color=c, label=f"Area {a}: {AREA_NAMES[a]}")
                  for a, c in zip(AREAS, AREA_COLORS)]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=8, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(data_path("explore_4_spatial_features.png"), bbox_inches="tight")
plt.show()
print("[4] Saved explore_4_spatial_features.png")

# ================================================================
# Figure 5: Correlation heatmap + scatter plot colored by area
# ================================================================
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 2, wspace=0.35)

# 5a. Numerical feature correlation heatmap
ax_corr = fig.add_subplot(gs[0, 0])
corr = df[NUM_COLS].corr()
im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax_corr, fraction=0.046, label="Pearson r")
ax_corr.set_xticks(range(len(NUM_COLS)))
ax_corr.set_yticks(range(len(NUM_COLS)))
short_names = ["Elev", "Aspect", "Slope", "H_Hydro", "V_Hydro",
               "H_Road", "Shade9", "ShadeN", "Shade3", "H_Fire"]
ax_corr.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax_corr.set_yticklabels(short_names, fontsize=8)
ax_corr.set_title("Numerical Feature Correlation Heatmap", fontsize=9)
for i in range(len(NUM_COLS)):
    for j in range(len(NUM_COLS)):
        val = corr.iloc[i, j]
        ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color="white" if abs(val) > 0.6 else "black")

# 5b. Elevation vs Distance_To_Roadways, colored by area
ax_sc = fig.add_subplot(gs[0, 1])
sample = df.sample(min(3000, len(df)), random_state=42)
for area, c in zip(AREAS, AREA_COLORS):
    sub = sample[sample[SPATIAL_COL] == area]
    ax_sc.scatter(sub["Elevation"],
                  sub["Horizontal_Distance_To_Roadways"],
                  c=c, s=8, alpha=0.5, label=f"Area {area}: {AREA_NAMES[area]}")
ax_sc.set_xlabel("Elevation")
ax_sc.set_ylabel("Horizontal_Distance_To_Roadways")
ax_sc.set_title("Elevation vs Distance to Roadways\nColored by Wilderness Area (spatial separation visible)",
                fontsize=9)
ax_sc.legend(fontsize=8, markerscale=2)
ax_sc.grid(alpha=0.25)

fig.suptitle("Feature Correlation & Spatial Separation Scatter", fontsize=11, fontweight="bold")
plt.savefig(data_path("explore_5_corr_scatter.png"), bbox_inches="tight")
plt.show()
print("[5] Saved explore_5_corr_scatter.png")

# -- Console summary -----------------------------------------------------------
print("\n" + "=" * 60)
print("Data Summary".center(60))
print("=" * 60)
print(f"\nTotal samples: {len(df)}")
print(f"Columns: {df.shape[1]} (10 numerical + Wilderness_Area + Soil_Type + Cover_Type)")
print(f"\nNumerical feature statistics:")
print(df[NUM_COLS].agg(["mean", "std", "min", "max"]).round(1).to_string())
print(f"\nDominant Cover_Type per area:")
for area in AREAS:
    sub = df[df[SPATIAL_COL] == area]
    dominant = sub[LABEL_COL].value_counts().idxmax()
    pct = sub[LABEL_COL].value_counts(normalize=True).max()
    print(f"  Area {area} {AREA_NAMES[area]:18s}: {len(sub):5d} rows  "
          f"dominant=Type {dominant}({type_desc[dominant]}) {pct:.1%}")
print("\nDone. 5 figures saved to current directory.")
