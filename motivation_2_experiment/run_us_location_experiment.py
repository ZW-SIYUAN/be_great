"""
run_us_location_experiment.py
==============================
FD discovery + noise-injection robustness experiment on the US Location dataset.

Dataset
-------
  motivation_2_experiment/us_location/data/us_location_train.csv
  16,320 rows × 5 columns:
    state_code  (51 unique)  – US state / territory abbreviation
    lat         (2,573 unique) – latitude  (2 d.p.)
    lon         (5,473 unique) – longitude (2 d.p.)
    bird        (30 unique)  – state bird
    lat_zone    (3 unique)   – {low, middle, high}

Ground-truth FDs (4 minimal, semantically grounded)
----------------------------------------------------
  1. {state_code}  →  bird
       Every US state has exactly one state bird.
  2. {lat}         →  lat_zone
       Latitude uniquely determines the latitudinal zone.
  3. {lat, lon}    →  state_code
       A GPS coordinate belongs to exactly one state.
  4. {lat, lon}    →  bird
       By transitivity: (lat,lon) → state_code → bird.
       Minimal because neither lat alone nor lon alone determines bird.

  NOTE: {lat, lon} → lat_zone holds but is NOT minimal
        ({lat} alone already determines lat_zone), so the exact algorithms
        correctly omit it and it is excluded from the ground truth.

Pipeline
--------
  Phase 0 – Clean baseline (0 % noise)
  Phase 1 – Noise injection at 5 / 10 / 15 / 20 / 25 / 30 %
    For each level p:
      1. Corrupt p% of cells per column (cell-level, per-column, independent).
      2. Discover FDs with HyFD and TANE.
      3. Classify against the 4 ground-truth FDs.
      4. Compute Precision, Recall, F1.
      5. Persist noisy dataset + per-level results CSV.
  Phase 2 – Summary CSV + F1-score bar chart (0 %–30 %).

Outputs  →  motivation_2_experiment/us_location/results/
  fd_results_clean.csv                 baseline FD detail (0 % noise)
  noisy_05pct.csv … noisy_30pct.csv   six noisy datasets
  fd_results_05pct.csv … fd_results_30pct.csv  per-level FD detail
  summary.csv                          aggregated metrics
  us_location_plot.png                 F1-score grouped bar chart
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from fd_algorithms import hyfd, tane          # noqa: E402
from run_fd_discovery import (                # noqa: E402
    AlgoResult,
    FD,
    run_algorithm,
    save_results,
    verify_fd,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = _HERE / "us_location" / "data" / "us_location_train.csv"
OUT_DIR   = _HERE / "us_location" / "results"

# ── Ground-truth FDs for US Location dataset ──────────────────────────────────
GROUND_TRUTH: list[tuple[frozenset[str], str]] = [
    (frozenset({"state_code"}),       "bird"),        # 1. state bird
    (frozenset({"lat"}),              "lat_zone"),    # 2. latitude zone
    (frozenset({"lat", "lon"}),       "state_code"),  # 3. GPS → state
    (frozenset({"lat", "lon"}),       "bird"),        # 4. GPS → bird (transitive)
]
GT_SET = {(lhs, rhs) for lhs, rhs in GROUND_TRUTH}
N_GT   = len(GROUND_TRUTH)

# ── Experiment parameters ─────────────────────────────────────────────────────
NOISE_LEVELS   : list[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SEED_BASE      : int          = 200     # seed for level i  →  SEED_BASE + i
N_SAMPLE_PAIRS : int          = 60
HYFD_SEED      : int          = 0

# ── Separators ────────────────────────────────────────────────────────────────
_SEP  = "=" * 72
_DASH = "-" * 72


# ── FD classification (custom GT) ─────────────────────────────────────────────

def classify_wrt_gt(
    discovered: list[FD],
    df: pd.DataFrame,
) -> tuple[list[FD], list[FD], list[FD], list[tuple[frozenset[str], str]]]:
    """
    Classify discovered FDs against the US Location ground truth.

    Categories
    ----------
    designed_hit   – present in GROUND_TRUTH
    implicit_valid – not in GT but verifiably holds in df
    false_positive – reported but does NOT hold (should be empty for exact algos)
    missed_gt      – GT FDs absent from the discovered set

    Returns
    -------
    (designed_hits, implicit_valid, false_positives, missed_gt)
    """
    discovered_keys = {(fd.lhs, fd.rhs) for fd in discovered}
    designed_hits:   list[FD] = []
    implicit_valid:  list[FD] = []
    false_positives: list[FD] = []

    for fd in discovered:
        key = (fd.lhs, fd.rhs)
        if key in GT_SET:
            designed_hits.append(fd)
        elif verify_fd(df, fd.lhs, fd.rhs):
            implicit_valid.append(fd)
        else:
            false_positives.append(fd)

    missed_gt = [
        (lhs, rhs)
        for lhs, rhs in GROUND_TRUTH
        if (lhs, rhs) not in discovered_keys
    ]
    return designed_hits, implicit_valid, false_positives, missed_gt


# ── Noise injection ────────────────────────────────────────────────────────────

def inject_noise(df: pd.DataFrame, noise_rate: float, seed: int) -> pd.DataFrame:
    """
    Cell-level random noise injection (per-column, independent).

    For every column c, randomly select floor(noise_rate × n) distinct row
    indices and overwrite those cells with values drawn uniformly from c's
    observed value domain.

    Parameters
    ----------
    df         : clean source DataFrame (never modified in-place).
    noise_rate : fraction of rows per column to corrupt (e.g. 0.05 → 5 %).
    seed       : RNG seed for reproducibility.
    """
    rng       = np.random.default_rng(seed)
    df_noisy  = df.copy()
    n         = len(df)
    n_corrupt = max(1, int(round(noise_rate * n)))

    for j, col in enumerate(df.columns):
        domain   = df[col].to_numpy()
        idx      = rng.choice(n, size=n_corrupt, replace=False)
        new_vals = rng.choice(domain, size=n_corrupt)
        df_noisy.iloc[idx, j] = new_vals

    return df_noisy


# ── Per-level runner ───────────────────────────────────────────────────────────

def run_one_level(df: pd.DataFrame) -> list[AlgoResult]:
    """Run HyFD and TANE on *df*; return one AlgoResult per algorithm."""
    registry = [
        ("HyFD", hyfd.discover, {"n_sample_pairs": N_SAMPLE_PAIRS, "seed": HYFD_SEED}),
        ("TANE", tane.discover, {}),
    ]
    results: list[AlgoResult] = []
    for name, fn, kwargs in registry:
        fds, elapsed, stats = run_algorithm(fn, df, kwargs)
        designed, implicit, fp, missed = classify_wrt_gt(fds, df)
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


# ── PRF computation (uses local N_GT, not run_fd_discovery.GROUND_TRUTH) ──────

def _prf(r: AlgoResult) -> tuple[float, float, float]:
    """
    Compute Precision, Recall, F1 w.r.t. the *us_location* ground truth (N_GT=4).

    AlgoResult.recall uses len(run_fd_discovery.GROUND_TRUTH) = 7, which is
    incorrect for this dataset.  We recompute all three metrics here.

    precision = hits / (hits + fp)
    recall    = hits / N_GT          ← N_GT = 4 for us_location
    f1        = harmonic mean(P, R)
    """
    hits = r.n_designed_hits
    fp   = r.n_fp
    prec = hits / (hits + fp) if (hits + fp) > 0 else 0.0
    rec  = hits / N_GT
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


# ── Summary row builder ────────────────────────────────────────────────────────

def _result_to_row(r: AlgoResult, noise_pct: int) -> dict:
    prec, rec, f1 = _prf(r)
    s = r.algo_stats
    return {
        "noise_pct":             noise_pct,
        "algorithm":             r.name,
        "total_fds":             len(r.discovered),
        "gt_hits":               r.n_designed_hits,
        "implicit_valid":        r.n_implicit,
        "false_positives":       r.n_fp,
        "false_negatives":       r.n_missed,
        "precision":             prec,
        "recall":                rec,
        "f1":                    f1,
        "n_fd_checks":           s.get("n_fd_checks",             ""),
        "n_sp_products":         s.get("n_sp_products",           ""),
        "n_keys_pruned":         s.get("n_keys_pruned",           ""),
        "n_refuted_by_sampling": s.get("n_refuted_by_sampling",   "—"),
        "wall_time_s":           round(r.total_s, 3),
    }


# ── Console reporting ──────────────────────────────────────────────────────────

def _print_level_table(results: list[AlgoResult], noise_pct: int | str) -> None:
    label = f"{noise_pct}%  (clean)" if noise_pct == 0 else f"{noise_pct}%"
    names = [r.name for r in results]
    col_w = 14

    print(f"\n  Noise = {label}  ──  Head-to-Head Comparison")
    header = f"  {'Metric':<44}" + "".join(f"  {n:>{col_w}}" for n in names)
    print(header)
    print(f"  {'-' * (44 + (col_w + 2) * len(results))}")

    def _row(lbl: str, vals: list) -> str:
        return f"  {lbl:<44}" + "".join(f"  {str(v):>{col_w}}" for v in vals)

    rows: list[tuple[str, list]] = [
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
            [f"{_prf(r)[0]:.4f}" for r in results]),
        ("Recall",
            [f"{_prf(r)[1]:.4f}" for r in results]),
        ("F1-score",
            [f"{_prf(r)[2]:.4f}" for r in results]),
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
    for lbl, vals in rows:
        print(_row(lbl, vals))
    print()


def _print_discovered_fds(results: list[AlgoResult]) -> None:
    """Print the categorised FD lists for each algorithm."""
    for r in results:
        print(f"  [{r.name}]  discovered {len(r.discovered)} FDs:")
        if r.designed_hits:
            print(f"    [OK] Designed GT hits ({len(r.designed_hits)}):")
            for fd in sorted(r.designed_hits, key=lambda f: (len(f.lhs), sorted(f.lhs), f.rhs)):
                print(f"           {fd}")
        if r.implicit_valid:
            print(f"    [~~] Implicit valid ({len(r.implicit_valid)}):")
            for fd in sorted(r.implicit_valid, key=lambda f: (len(f.lhs), sorted(f.lhs), f.rhs)):
                print(f"           {fd}")
        if r.missed_gt:
            print(f"    [--] Missed GT FDs ({len(r.missed_gt)}):")
            for lhs, rhs in r.missed_gt:
                lhs_str = ", ".join(sorted(lhs))
                print(f"           {{{lhs_str}}} -> {rhs}")
        print()


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(summary_df: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped bar chart.

    X-axis  : noise injection rate (0 % clean baseline + 5 %–30 %)
    Left Y  : F1-score bars + dashed trend lines for HyFD and TANE.
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
    C_H    = "#1565C0"   # HyFD bar  (Material Blue 800)
    C_T    = "#BF360C"   # TANE bar  (Material Deep-Orange 900)
    C_H_TR = "#64B5F6"   # HyFD trend (Blue 300)
    C_T_TR = "#FFAB40"   # TANE trend (Amber 400)

    bar_w = 0.30
    fig, ax = plt.subplots(figsize=(12, 6))

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

    # ── Trend lines ───────────────────────────────────────────────────────────
    ax.plot(x - bar_w / 2, hyfd_f1, "o--",
            color=C_H_TR, linewidth=1.6, markersize=5.5,
            zorder=4, alpha=0.90)
    ax.plot(x + bar_w / 2, tane_f1, "o--",
            color=C_T_TR, linewidth=1.6, markersize=5.5,
            zorder=4, alpha=0.90)

    # ── Value labels ──────────────────────────────────────────────────────────
    for bars, col in ((bars_h, C_H), (bars_t, C_T)):
        for bar in bars:
            h = bar.get_height()
            txt = f"{h:.3f}" if h >= 5e-4 else "0.000"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.020, txt,
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=col,
            )

    # ── X-axis tick labels (highlight the clean baseline) ─────────────────────
    tick_labels = []
    for p in noise_pcts:
        tick_labels.append("0%\n(clean)" if p == 0 else f"{p}%")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel("Noise injection rate", fontsize=12, labelpad=6)
    ax.set_ylabel(
        f"F1-Score  (w.r.t. {N_GT} ground-truth FDs)",
        fontsize=12, labelpad=6,
    )
    ax.set_title(
        "Robustness of Exact FD Discovery Under Cell-Level Noise Injection\n"
        "US Location Dataset  ·  n = 16,320 rows  ·  5 columns  "
        f"·  {N_GT} ground-truth FDs",
        fontsize=12, pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylim(0.0, 1.25)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.45, zorder=0)
    ax.grid(axis="y", which="minor", linestyle=":",  alpha=0.20, zorder=0)
    ax.legend(fontsize=11, loc="upper right",
              framealpha=0.93, edgecolor="#cccccc")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Shade 0% column to distinguish clean baseline ─────────────────────────
    ax.axvspan(-0.5, 0.5, color="#E3F2FD", alpha=0.35, zorder=0)
    ax.text(0, 1.18, "clean\nbaseline", ha="center", fontsize=8,
            color="#1565C0", style="italic")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved  ->  {out_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        sys.exit(
            f"[ERROR] Dataset not found: {DATA_PATH}\n"
            f"        Expected: us_location/data/us_location_train.csv"
        )

    df_clean = pd.read_csv(DATA_PATH)
    n_rows, n_cols = df_clean.shape

    print(_SEP)
    print("  US Location FD Discovery + Noise Experiment")
    print(_SEP)
    print(f"  Dataset   : {DATA_PATH}")
    print(f"  Shape     : {n_rows:,} rows x {n_cols} columns")
    print(f"  Columns   : {list(df_clean.columns)}")
    print(f"  GT FDs    : {N_GT}")
    print(f"  Output    : {OUT_DIR}")
    print()
    print("  Ground-truth FDs:")
    for i, (lhs, rhs) in enumerate(GROUND_TRUTH, 1):
        lhs_str = ", ".join(sorted(lhs))
        print(f"    {i}. {{{lhs_str}}} -> {rhs}")
    print()

    all_rows: list[dict] = []

    # ── Phase 0: Clean baseline ────────────────────────────────────────────────
    print(_SEP)
    print("  PHASE 0  ──  Clean baseline (0% noise)")
    print(_SEP)
    print("  Running HyFD & TANE on clean data ...", flush=True)

    t0 = time.perf_counter()
    clean_results = run_one_level(df_clean)
    print(f"  Baseline total time  :  {time.perf_counter() - t0:.2f} s")

    _print_level_table(clean_results, 0)
    _print_discovered_fds(clean_results)

    clean_fd_path = OUT_DIR / "fd_results_clean.csv"
    save_results(clean_results, clean_fd_path)
    print(f"  Clean FD results saved  ->  {clean_fd_path}\n")

    for r in clean_results:
        all_rows.append(_result_to_row(r, 0))

    # ── Phase 1: Noise injection ───────────────────────────────────────────────
    for i, noise_rate in enumerate(NOISE_LEVELS):
        pct  = int(round(noise_rate * 100))
        seed = SEED_BASE + i

        print(_SEP)
        print(f"  NOISE LEVEL  {pct:2d}%   "
              f"(n_corrupt = {int(round(noise_rate * n_rows)):,} cells/col, seed = {seed})")
        print(_SEP)

        # 1. Generate noisy dataset
        df_noisy   = inject_noise(df_clean, noise_rate, seed=seed)
        noisy_path = OUT_DIR / f"noisy_{pct:02d}pct.csv"
        df_noisy.to_csv(noisy_path, index=False)
        print(f"  Noisy dataset saved     ->  {noisy_path}")

        # 2. Discover FDs
        print(f"  Running HyFD & TANE ...", flush=True)
        t_level  = time.perf_counter()
        results  = run_one_level(df_noisy)
        print(f"  Level total time        :   {time.perf_counter() - t_level:.2f} s")

        # 3. Console report
        _print_level_table(results, pct)

        # 4. Persist per-level FD detail CSV
        fd_path = OUT_DIR / f"fd_results_{pct:02d}pct.csv"
        save_results(results, fd_path)
        print(f"  FD results saved        ->  {fd_path}\n")

        for r in results:
            all_rows.append(_result_to_row(r, pct))

    # ── Phase 2: Aggregated summary ────────────────────────────────────────────
    summary_df   = pd.DataFrame(all_rows)
    summary_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved  ->  {summary_path}")

    # ── Final table to console ─────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print(f"  {'Noise':>7}  {'Algo':>6}  {'FDs':>5}  {'Hits':>5}  "
          f"{'P':>7}  {'R':>7}  {'F1':>7}  {'Time(s)':>8}")
    print(f"  {_DASH}")
    for row in all_rows:
        noise_lbl = f"{row['noise_pct']}%" if row["noise_pct"] > 0 else "0% (clean)"
        print(
            f"  {noise_lbl:>10}  "
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
    plot_path = OUT_DIR / "us_location_plot.png"
    plot_results(summary_df, plot_path)


if __name__ == "__main__":
    main()
