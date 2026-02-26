"""
run_noise_experiment.py
=======================
Noise-injection robustness experiment for exact FD discovery algorithms.

Pipeline
--------
  For each noise level p ∈ {5, 10, 15, 20, 25, 30}%:
    1. Load the clean synthetic FD benchmark dataset.
    2. Inject p% cell-level random noise into a fresh copy of the dataset.
    3. Discover FDs with HyFD and TANE on the noisy dataset.
    4. Classify each FD as designed_hit / implicit_valid / false_positive.
    5. Compute Precision, Recall, F1 w.r.t. the 7 designed ground-truth FDs.
    6. Save the noisy dataset and per-level FD results to CSV.
  Aggregate a summary CSV and render the F1-score grouped bar chart.

Noise injection model
---------------------
  Cell-level, per-column, independent:
    For every column c, randomly select floor(p × n) distinct row indices
    (without replacement) and replace those cells with values drawn uniformly
    (with replacement) from c's observed value domain.
    Expected total corruption ≈ p% of all cells.

Outputs  →  motivation_2_experiment/noise_results/
  noisy_05pct.csv  …  noisy_30pct.csv      six noisy datasets (13 cols, 10k rows)
  fd_results_05pct.csv … fd_results_30pct.csv  per-level FD discovery detail
  summary.csv                               aggregated metrics for all levels
  noise_experiment_plot.png                 F1-score grouped bar chart
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from fd_algorithms import hyfd, tane             # noqa: E402
from run_fd_discovery import (                    # noqa: E402
    GROUND_TRUTH,
    AlgoResult,
    FD,
    classify,
    run_algorithm,
    save_results,
)

# ── Configuration ─────────────────────────────────────────────────────────────
NOISE_LEVELS   : list[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SEED_BASE      : int          = 100     # seed for noise level i  →  SEED_BASE + i
N_SAMPLE_PAIRS : int          = 60      # HyFD Phase-1 budget
HYFD_SEED      : int          = 0       # HyFD internal RNG seed

N_GT           : int = len(GROUND_TRUTH)

DATASET_PATH   : Path = _HERE / "synthetic_fd_dataset.csv"
OUT_DIR        : Path = _HERE / "noise_results"

# ── Noise injection ────────────────────────────────────────────────────────────

def inject_noise(df: pd.DataFrame, noise_rate: float, seed: int) -> pd.DataFrame:
    """
    Cell-level random noise injection (per-column, independent).

    For every column c, randomly draw ``floor(noise_rate × n)`` distinct row
    indices (without replacement) and overwrite those cells with values sampled
    uniformly (with replacement) from c's observed value domain.

    Parameters
    ----------
    df : pd.DataFrame
        Clean source DataFrame; never modified in-place.
    noise_rate : float
        Fraction of rows per column to corrupt (e.g. 0.05 for 5 %).
    seed : int
        RNG seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        A fresh copy of *df* with the requested noise applied.
    """
    rng       = np.random.default_rng(seed)
    df_noisy  = df.copy()
    n         = len(df)
    n_corrupt = max(1, int(round(noise_rate * n)))

    for j, col in enumerate(df.columns):
        domain   = df[col].to_numpy()                              # observed values
        idx      = rng.choice(n, size=n_corrupt, replace=False)    # rows to corrupt
        new_vals = rng.choice(domain, size=n_corrupt)              # replacement values
        df_noisy.iloc[idx, j] = new_vals

    return df_noisy


# ── Per-level FD runner ────────────────────────────────────────────────────────

def run_one_level(df: pd.DataFrame) -> list[AlgoResult]:
    """
    Run HyFD and TANE on *df*; return one :class:`AlgoResult` per algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset (clean or noisy) to analyse.

    Returns
    -------
    list[AlgoResult]
        Two elements: index 0 = HyFD, index 1 = TANE.
    """
    registry = [
        ("HyFD", hyfd.discover, {"n_sample_pairs": N_SAMPLE_PAIRS, "seed": HYFD_SEED}),
        ("TANE", tane.discover, {}),
    ]
    results: list[AlgoResult] = []
    for name, fn, kwargs in registry:
        fds, elapsed, stats = run_algorithm(fn, df, kwargs)
        designed, implicit, fp, missed = classify(fds, df)
        results.append(AlgoResult(
            name            = name,
            discovered      = fds,
            designed_hits   = designed,
            implicit_valid  = implicit,
            false_positives = fp,
            missed_gt       = missed,
            algo_stats      = stats,
            total_s         = elapsed,
        ))
    return results


# ── Summary row builder ────────────────────────────────────────────────────────

def _result_to_row(r: AlgoResult, noise_pct: int) -> dict:
    """Flatten one AlgoResult into a flat dict for the summary CSV."""
    s = r.algo_stats
    return {
        "noise_pct":             noise_pct,
        "algorithm":             r.name,
        "total_fds":             len(r.discovered),
        "gt_hits":               r.n_designed_hits,
        "implicit_valid":        r.n_implicit,
        "false_positives":       r.n_fp,
        "false_negatives":       r.n_missed,
        "precision":             round(r.precision, 4),
        "recall":                round(r.recall, 4),
        "f1":                    round(r.f1, 4),
        "n_fd_checks":           s.get("n_fd_checks",             ""),
        "n_sp_products":         s.get("n_sp_products",           ""),
        "n_keys_pruned":         s.get("n_keys_pruned",           ""),
        "n_refuted_by_sampling": s.get("n_refuted_by_sampling",   "—"),
        "wall_time_s":           round(r.total_s, 3),
    }


# ── Console reporting ──────────────────────────────────────────────────────────

_SEP  = "=" * 72
_DASH = "-" * 72


def _print_level_table(results: list[AlgoResult], noise_pct: int) -> None:
    """
    Print the same wide-format comparison table used in run_fd_discovery.py,
    annotated with the current noise level.
    """
    names  = [r.name for r in results]
    col_w  = 14

    print(f"\n  Noise = {noise_pct}%  ──  Head-to-Head Comparison")
    header = f"  {'Metric':<42}" + "".join(f"  {n:>{col_w}}" for n in names)
    print(header)
    print(f"  {'-' * (42 + (col_w + 2) * len(results))}")

    def _row(label: str, vals: list) -> str:
        return f"  {label:<42}" + "".join(f"  {str(v):>{col_w}}" for v in vals)

    metrics: list[tuple[str, list]] = [
        ("Total FDs discovered",
            [len(r.discovered) for r in results]),
        (f"Ground-truth hits (out of {N_GT})",
            [f"{r.n_designed_hits} / {N_GT}" for r in results]),
        ("Implicit valid FDs",
            [r.n_implicit for r in results]),
        ("False positives",
            [r.n_fp for r in results]),
        ("False negatives",
            [r.n_missed for r in results]),
        ("Precision",
            [f"{r.precision:.4f}" for r in results]),
        ("Recall",
            [f"{r.recall:.4f}" for r in results]),
        ("F1-score",
            [f"{r.f1:.4f}" for r in results]),
        ("-- Internal counters --",
            [""] * len(results)),
        ("PLI fd_holds_with calls",
            [r.algo_stats.get("n_fd_checks",            "-") for r in results]),
        ("Stripped-partition products",
            [r.algo_stats.get("n_sp_products",          "-") for r in results]),
        ("Candidates pruned (superkeys)",
            [r.algo_stats.get("n_keys_pruned",          "-") for r in results]),
        ("Refuted by sampling (Phase 1)",
            [r.algo_stats.get("n_refuted_by_sampling",  "—") for r in results]),
        ("Wall-clock time (s)",
            [f"{r.total_s:.2f}" for r in results]),
    ]
    for label, vals in metrics:
        print(_row(label, vals))
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(summary_df: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped bar chart: X-axis = noise injection rate, left Y-axis = F1-score.

    Two bar groups per noise level (HyFD and TANE) with dashed trend lines
    connecting bar tops.  A horizontal reference line at F1 = 1.0 marks the
    clean-data baseline.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Aggregated metrics with columns ``noise_pct``, ``algorithm``, ``f1``,
        ``precision``, ``recall``.
    out_path : Path
        File path for the output PNG.
    """
    noise_pcts = sorted(summary_df["noise_pct"].unique())
    x = np.arange(len(noise_pcts))

    def _col(algo: str, metric: str) -> np.ndarray:
        return np.array([
            float(summary_df.loc[
                (summary_df["algorithm"] == algo) &
                (summary_df["noise_pct"] == p),
                metric
            ].values[0])
            for p in noise_pcts
        ])

    hyfd_f1 = _col("HyFD", "f1")
    tane_f1 = _col("TANE", "f1")

    # ── Colours ───────────────────────────────────────────────────────────────
    C_H     = "#1565C0"   # HyFD bar fill  (Material Blue 800)
    C_T     = "#BF360C"   # TANE bar fill  (Material Deep-Orange 900)
    C_H_TR  = "#64B5F6"   # HyFD trend line (Blue 300)
    C_T_TR  = "#FFAB40"   # TANE trend line (Amber 400)

    bar_w = 0.30
    fig, ax = plt.subplots(figsize=(11, 6))

    # ── Bars ──────────────────────────────────────────────────────────────────
    bars_h = ax.bar(
        x - bar_w / 2, hyfd_f1, bar_w,
        label="HyFD — F1-Score",
        color=C_H, alpha=0.87, zorder=3,
        edgecolor="white", linewidth=0.6,
    )
    bars_t = ax.bar(
        x + bar_w / 2, tane_f1, bar_w,
        label="TANE — F1-Score",
        color=C_T, alpha=0.87, zorder=3,
        edgecolor="white", linewidth=0.6,
    )

    # ── Trend lines connecting bar tops ───────────────────────────────────────
    ax.plot(
        x - bar_w / 2, hyfd_f1, "o--",
        color=C_H_TR, linewidth=1.6, markersize=5.5,
        zorder=4, alpha=0.90,
    )
    ax.plot(
        x + bar_w / 2, tane_f1, "o--",
        color=C_T_TR, linewidth=1.6, markersize=5.5,
        zorder=4, alpha=0.90,
    )

    # ── Value labels on bar tops ───────────────────────────────────────────────
    for bars, col in ((bars_h, C_H), (bars_t, C_T)):
        for bar in bars:
            h = bar.get_height()
            txt = f"{h:.3f}" if h >= 5e-4 else "0.000"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.020,
                txt,
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=col,
            )

    # ── Clean baseline reference ───────────────────────────────────────────────
    ax.axhline(
        y=1.0, color="#616161", linewidth=1.1,
        linestyle=":", alpha=0.60, zorder=1,
    )
    ax.text(
        len(noise_pcts) - 0.52, 1.025,
        "clean baseline  F1 = 1.000  (0 % noise)",
        fontsize=8, color="#616161", ha="right", style="italic",
    )

    # ── Axes and grid ──────────────────────────────────────────────────────────
    ax.set_xlabel("Noise injection rate", fontsize=12, labelpad=6)
    ax.set_ylabel(
        "F1-Score  (w.r.t. 7 designed ground-truth FDs)",
        fontsize=12, labelpad=6,
    )
    ax.set_title(
        "Robustness of Exact FD Discovery Under Cell-Level Noise Injection\n"
        "Synthetic FD Benchmark  ·  n = 10,000 rows  ·  13 columns  "
        "·  7 ground-truth FDs",
        fontsize=12, pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in noise_pcts], fontsize=11)
    ax.set_ylim(0.0, 1.22)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.45, zorder=0)
    ax.grid(axis="y", which="minor", linestyle=":",  alpha=0.20, zorder=0)
    ax.legend(
        fontsize=11, loc="upper right",
        framealpha=0.93, edgecolor="#cccccc",
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved  ->  {out_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        sys.exit(
            f"[ERROR] Dataset not found: {DATASET_PATH}\n"
            f"        Run  python generate_dataset.py  first."
        )

    print(f"Dataset  : {DATASET_PATH}")
    df_clean = pd.read_csv(DATASET_PATH)
    print(f"  Shape        : {len(df_clean):,} rows x {len(df_clean.columns)} cols")
    print(f"  Ground truth : {N_GT} designed FDs")
    print(f"  Output dir   : {OUT_DIR}\n")

    all_rows: list[dict] = []

    # ── Iterate over all noise levels ─────────────────────────────────────────
    for i, noise_rate in enumerate(NOISE_LEVELS):
        pct  = int(round(noise_rate * 100))
        seed = SEED_BASE + i

        print(_SEP)
        print(f"  NOISE LEVEL  {pct:2d}%   (n_corrupt = {int(round(noise_rate * len(df_clean))):,} cells/col, seed = {seed})")
        print(_SEP)

        # 1. Generate noisy dataset
        df_noisy   = inject_noise(df_clean, noise_rate, seed=seed)
        noisy_path = OUT_DIR / f"noisy_{pct:02d}pct.csv"
        df_noisy.to_csv(noisy_path, index=False)
        print(f"  Noisy dataset saved    ->  {noisy_path}")

        # 2. Discover FDs (HyFD + TANE)
        print(f"  Running HyFD & TANE ...", flush=True)
        t_level = time.perf_counter()
        results  = run_one_level(df_noisy)
        print(f"  Level total time       :   {time.perf_counter() - t_level:.2f} s")

        # 3. Console report (wide format)
        _print_level_table(results, pct)

        # 4. Save per-level FD detail CSV (same schema as fd_benchmark_results.csv)
        fd_path = OUT_DIR / f"fd_results_{pct:02d}pct.csv"
        save_results(results, fd_path)
        print(f"  FD results saved       ->  {fd_path}\n")

        # 5. Accumulate summary rows
        for r in results:
            all_rows.append(_result_to_row(r, pct))

    # ── Aggregated summary CSV ─────────────────────────────────────────────────
    summary_df   = pd.DataFrame(all_rows)
    summary_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved  ->  {summary_path}")

    # ── Final table to console ─────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print(f"  {'Noise':>6}  {'Algo':>6}  {'FDs':>5}  {'Hits':>5}  "
          f"{'P':>7}  {'R':>7}  {'F1':>7}  {'Time(s)':>8}")
    print(f"  {_DASH}")
    for row in all_rows:
        print(
            f"  {row['noise_pct']:>5}%  "
            f"{row['algorithm']:>6}  "
            f"{row['total_fds']:>5}  "
            f"{row['gt_hits']:>3}/{N_GT}  "
            f"{row['precision']:>7.4f}  "
            f"{row['recall']:>7.4f}  "
            f"{row['f1']:>7.4f}  "
            f"{row['wall_time_s']:>8.2f}"
        )
    print(f"{_SEP}\n")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_path = OUT_DIR / "noise_experiment_plot.png"
    plot_results(summary_df, plot_path)


if __name__ == "__main__":
    main()
