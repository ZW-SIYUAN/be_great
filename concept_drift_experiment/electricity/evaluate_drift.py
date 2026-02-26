import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─── 数据加载 ────────────────────────────────────────────────────
real  = pd.read_csv("electricity.csv").iloc[:2000].reset_index(drop=True)
synth = pd.read_csv("electricity_synthetic.csv").iloc[:2000].reset_index(drop=True)

NUM_COLS = ["date", "day", "period", "nswprice", "nswdemand",
            "vicprice", "vicdemand", "transfer"]
T = np.arange(len(real))

for df in [real, synth]:
    df["label"] = (df["class"] == "UP").astype(int)

# ─── 全局样式 ────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
REAL_COLOR, SYNTH_COLOR = "#1f77b4", "#ff7f0e"

# ════════════════════════════════════════════════════════════════
# 1. 多窗口滚动 P(UP)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, w in zip(axes, [50, 100, 200]):
    ax.plot(T, real["label"].rolling(w).mean(),
            color=REAL_COLOR, label="Real", lw=1.5)
    ax.plot(T, synth["label"].rolling(w).mean(),
            color=SYNTH_COLOR, label="Synthetic", lw=1.5, ls="--")
    ax.set_ylabel(f"P(UP)  w={w}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Time index")
fig.suptitle("Concept Drift: Rolling P(UP) — Real vs Synthetic", fontweight="bold")
plt.tight_layout()
plt.savefig("drift_1_rolling_pup.png")
plt.show()
print("[1] 已保存 drift_1_rolling_pup.png")

# ════════════════════════════════════════════════════════════════
# 2. 各数值特征的滚动均值（窗口=100）
# ════════════════════════════════════════════════════════════════
W = 100
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()
for ax, col in zip(axes, NUM_COLS):
    ax.plot(T, real[col].rolling(W).mean(),
            color=REAL_COLOR, label="Real", lw=1.2)
    ax.plot(T, synth[col].rolling(W).mean(),
            color=SYNTH_COLOR, label="Synthetic", lw=1.2, ls="--")
    ax.set_title(col)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle(f"Rolling Mean (w={W}) of Each Feature — Real vs Synthetic",
             fontweight="bold")
plt.tight_layout()
plt.savefig("drift_2_rolling_features.png")
plt.show()
print("[2] 已保存 drift_2_rolling_features.png")

# ════════════════════════════════════════════════════════════════
# 3. 滑动窗口 KS 检验（漂移强度曲线）
# ════════════════════════════════════════════════════════════════
def sliding_ks(series, win=100, step=20):
    """相邻两个窗口做 KS 检验，返回时间轴与 KS 序列。"""
    times, ks_vals = [], []
    for start in range(0, len(series) - 2 * win, step):
        w1 = series[start: start + win]
        w2 = series[start + win: start + 2 * win]
        ks, _ = stats.ks_2samp(w1, w2)
        times.append(start + win)
        ks_vals.append(ks)
    return np.array(times), np.array(ks_vals)

KEY_COLS = ["nswprice", "nswdemand", "vicprice", "transfer"]
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
axes = axes.flatten()
for ax, col in zip(axes, KEY_COLS):
    t_r, ks_r = sliding_ks(real[col].values)
    t_s, ks_s = sliding_ks(synth[col].values)
    ax.plot(t_r, ks_r, color=REAL_COLOR,   label="Real",      lw=1.5)
    ax.plot(t_s, ks_s, color=SYNTH_COLOR,  label="Synthetic", lw=1.5, ls="--")
    ax.set_title(f"KS drift: {col}")
    ax.set_xlabel("Time index")
    ax.set_ylabel("KS statistic")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("Sliding-Window KS Drift Intensity — Real vs Synthetic",
             fontweight="bold")
plt.tight_layout()
plt.savefig("drift_3_ks_intensity.png")
plt.show()
print("[3] 已保存 drift_3_ks_intensity.png")

# ════════════════════════════════════════════════════════════════
# 4. 分段分类准确率（概念漂移核心证据）
#    训练在第 i 段 → 测试在第 i+1 段
#    若 Real 与 Synth 的准确率走势相似，说明漂移模式被保留
# ════════════════════════════════════════════════════════════════
def segment_accuracy(df, n_seg=10, feature_cols=NUM_COLS, label_col="label"):
    size = len(df) // n_seg
    accs = []
    for i in range(n_seg - 1):
        train = df.iloc[i * size: (i + 1) * size]
        test  = df.iloc[(i + 1) * size: (i + 2) * size]
        X_tr  = train[feature_cols].fillna(0).values
        y_tr  = train[label_col].values
        X_te  = test[feature_cols].fillna(0).values
        y_te  = test[label_col].values
        if len(np.unique(y_tr)) < 2:
            accs.append(np.nan)
            continue
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))
    return accs

N_SEG = 10
acc_real  = segment_accuracy(real,  n_seg=N_SEG)
acc_synth = segment_accuracy(synth, n_seg=N_SEG)
seg_ids   = np.arange(1, N_SEG)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(seg_ids, acc_real,  "o-",  color=REAL_COLOR,  label="Real",      lw=2, ms=7)
ax.plot(seg_ids, acc_synth, "s--", color=SYNTH_COLOR, label="Synthetic", lw=2, ms=7)
ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random baseline (0.5)")
ax.set_xlabel("Segment index  (train on i → test on i+1)")
ax.set_ylabel("Accuracy")
ax.set_title("Concept Drift Evidence: Segment-to-Segment Classification Accuracy",
             fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("drift_4_segment_accuracy.png")
plt.show()
print("[4] 已保存 drift_4_segment_accuracy.png")

# ════════════════════════════════════════════════════════════════
# 5. 四分位时段分布对比
#    实线填充 = Real，虚线轮廓 = Synthetic
#    同色 = 同时段，不同颜色 = 不同时段
# ════════════════════════════════════════════════════════════════
N = len(real)
q_ranges = [(0, N//4), (N//4, N//2), (N//2, 3*N//4), (3*N//4, N)]
q_labels  = ["Q1 (0–25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 (75–100%)"]
colors_q  = ["#1a9850", "#91cf60", "#fc8d59", "#d73027"]

show_cols = ["label", "nswprice", "nswdemand"]
fig, axes = plt.subplots(len(show_cols), 1, figsize=(12, 9))
for ax, col in zip(axes, show_cols):
    for (s, e), lbl, c in zip(q_ranges, q_labels, colors_q):
        r_vals = real.iloc[s:e][col].dropna()
        s_vals = synth.iloc[s:e][col].dropna()
        lo = min(r_vals.min(), s_vals.min())
        hi = max(r_vals.max(), s_vals.max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(r_vals, bins=bins, density=True, alpha=0.4,
                color=c, label=f"Real {lbl}")
        ax.hist(s_vals, bins=bins, density=True, alpha=0.0,
                color=c, histtype="step", lw=2, ls="--",
                label=f"Synth {lbl}")
    col_name = "P(UP)" if col == "label" else col
    ax.set_title(f"Quartile Distribution: {col_name}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
fig.suptitle(
    "Temporal Distribution Shift by Quartile\n"
    "Real = filled bars  |  Synthetic = dashed outline  |  same color = same period",
    fontweight="bold"
)
plt.tight_layout()
plt.savefig("drift_5_quartile_dist.png")
plt.show()
print("[5] 已保存 drift_5_quartile_dist.png")

# ════════════════════════════════════════════════════════════════
# 6. 定量摘要
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("定量摘要".center(60))
print("=" * 60)

# 6a. 滚动 P(UP) 的 Pearson 相关（衡量漂移趋势的同步程度）
print("\n[A] 滚动 P(UP) 的时序相关系数（r 越接近 1 说明漂移趋势越吻合）")
for w in [50, 100, 200]:
    r_roll = real["label"].rolling(w).mean().dropna()
    s_roll = synth["label"].rolling(w).mean().dropna()
    n = min(len(r_roll), len(s_roll))
    rho, p = stats.pearsonr(r_roll.values[:n], s_roll.values[:n])
    print(f"  w={w:3d}  r={rho:+.4f}  p={p:.2e}")

# 6b. 各特征滚动均值的时序相关
print("\n[B] 各特征滚动均值（w=100）的时序相关系数")
for col in NUM_COLS:
    r_m = real[col].rolling(100).mean().dropna()
    s_m = synth[col].rolling(100).mean().dropna()
    n = min(len(r_m), len(s_m))
    rho, _ = stats.pearsonr(r_m.values[:n], s_m.values[:n])
    print(f"  {col:12s}  r={rho:+.4f}")

# 6c. 分段准确率统计
valid_pairs = [(r, s) for r, s in zip(acc_real, acc_synth)
               if not (np.isnan(r) or np.isnan(s))]
mae = np.mean([abs(r - s) for r, s in valid_pairs])
print(f"\n[C] 分段分类准确率")
print(f"  Real  均值: {np.nanmean(acc_real):.4f}  各段: {[f'{v:.3f}' if not np.isnan(v) else 'NaN' for v in acc_real]}")
print(f"  Synth 均值: {np.nanmean(acc_synth):.4f}  各段: {[f'{v:.3f}' if not np.isnan(v) else 'NaN' for v in acc_synth]}")
print(f"  MAE (Real vs Synth): {mae:.4f}  （越小说明合成数据的漂移程度越接近真实）")

# 6d. 全局 KS 检验（整体分布差异）
print("\n[D] 全局 KS 检验（stat 越小说明合成数据整体分布越接近真实）")
for col in NUM_COLS:
    ks, p = stats.ks_2samp(real[col].dropna(), synth[col].dropna())
    print(f"  {col:12s}  KS={ks:.4f}  p={p:.2e}")

print("\n" + "=" * 60)
print("评估完成。5 张图已保存至当前目录。")
