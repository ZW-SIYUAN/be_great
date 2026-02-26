"""
IB_experiment/us_location/Scatter_Plot_A.py
============================================
Figure 1b (Method A): GReaT Coordinate Validity -- Native API Joint Generation

Uses GReaT's built-in sample() API with start_col="state_code" to generate
rows where state_code is fixed to the target state and the remaining columns
(lat, lon, bird, lat_zone) are generated in **random order**, faithfully
mimicking GReaT's training distribution.

Key difference from Method B (Scatter_Plot.py):
  Method A (this): state_code first, rest in RANDOM column order → GReaT's
                   intended usage; model leverages learned joint distribution.
  Method B:        Fixed prompt "state_code is DE" with FIXED continuation order;
                   model must generate lat/lon first regardless of training order.

Method A gives a fairer picture of GReaT's actual FD capability because
it uses the same column-ordering randomisation seen during training.

Usage:
  python Scatter_Plot_A.py                        # 4 default states
  python Scatter_Plot_A.py --states DE FL WA NV   # custom state list
  python Scatter_Plot_A.py --n-samples 500        # more samples per state

Outputs (IB_experiment/us_location/):
  scatter_validity_A.png   -- 2x2 panel figure (paper Figure 1b, Method A)
  scatter_results_A.csv    -- per-state validity statistics
"""

import argparse
import json
import logging
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    from shapely.geometry import Point
    from shapely.geometry import shape as shapely_shape
except ImportError:
    print("ERROR: shapely is required.  Install with:  pip install shapely")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ROOT      = HERE.parent.parent
DATA_DIR  = HERE / "data"
MODEL_DIR = HERE / "us_great_model"
TRAIN_CSV = DATA_DIR / "us_location_train.csv"

sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_SAMPLES_PER_STATE  = 300   # target valid samples per state
OVERSAMPLE_FACTOR    = 1.5   # generate extra to account for NaN/parse failures
TEMPERATURE          = 0.7
MAX_LENGTH           = 200   # token budget per row (must cover all 5 columns)
RANDOM_SEED          = 42
DEFAULT_STATES       = ["DE", "RI", "FL", "NV"]

# ── State name / code mappings ────────────────────────────────────────────────
ABBREV_TO_NAME: Dict[str, str] = {
    "AL": "Alabama",        "AK": "Alaska",         "AZ": "Arizona",
    "AR": "Arkansas",       "CA": "California",     "CO": "Colorado",
    "CT": "Connecticut",    "DE": "Delaware",        "FL": "Florida",
    "GA": "Georgia",        "HI": "Hawaii",          "ID": "Idaho",
    "IL": "Illinois",       "IN": "Indiana",         "IA": "Iowa",
    "KS": "Kansas",         "KY": "Kentucky",        "LA": "Louisiana",
    "ME": "Maine",          "MD": "Maryland",        "MA": "Massachusetts",
    "MI": "Michigan",       "MN": "Minnesota",       "MS": "Mississippi",
    "MO": "Missouri",       "MT": "Montana",         "NE": "Nebraska",
    "NV": "Nevada",         "NH": "New Hampshire",   "NJ": "New Jersey",
    "NM": "New Mexico",     "NY": "New York",        "NC": "North Carolina",
    "ND": "North Dakota",   "OH": "Ohio",            "OK": "Oklahoma",
    "OR": "Oregon",         "PA": "Pennsylvania",    "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota",    "TN": "Tennessee",
    "TX": "Texas",          "UT": "Utah",            "VT": "Vermont",
    "VA": "Virginia",       "WA": "Washington",      "WV": "West Virginia",
    "WI": "Wisconsin",      "WY": "Wyoming",         "DC": "District of Columbia",
}
NAME_TO_ABBREV: Dict[str, str] = {v: k for k, v in ABBREV_TO_NAME.items()}

_GEOJSON_URL   = (
    "https://raw.githubusercontent.com/python-visualization/"
    "folium/master/examples/data/us-states.json"
)
_GEOJSON_CACHE = HERE / "us_states_boundaries.json"


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 – State boundary loading  (identical to Scatter_Plot.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_state_geometries(target_abbrevs: List[str]) -> Dict[str, object]:
    if not _GEOJSON_CACHE.exists():
        log.info(f"Downloading US state boundaries ...")
        try:
            urllib.request.urlretrieve(_GEOJSON_URL, _GEOJSON_CACHE)
            log.info(f"Cached to {_GEOJSON_CACHE}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download state boundaries: {exc}\n"
                f"Place us-states.json at {_GEOJSON_CACHE}"
            ) from exc

    with open(_GEOJSON_CACHE, encoding="utf-8") as fh:
        geojson = json.load(fh)

    geometries: Dict[str, object] = {}
    for feature in geojson["features"]:
        name   = feature["properties"].get("name", "")
        abbrev = NAME_TO_ABBREV.get(name)
        if abbrev in target_abbrevs:
            geometries[abbrev] = shapely_shape(feature["geometry"])
            log.info(f"  Loaded boundary: {name} ({abbrev})")

    missing = set(target_abbrevs) - set(geometries)
    if missing:
        log.warning(f"No boundary data for: {missing}")
    return geometries


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 – GReaT native generation (Method A)
# ══════════════════════════════════════════════════════════════════════════════

def sample_via_great_api(
    great_model,
    state_val:   str,
    n_samples:   int,
    device:      str,
    temperature: float = TEMPERATURE,
    max_length:  int   = MAX_LENGTH,
) -> Tuple[List[float], List[float]]:
    """Generate (lat, lon) pairs using GReaT's native sample() API.

    Calls great_model.sample() with start_col="state_code" so that:
      1. The generation starts with "state_code is <state_val>"
      2. Remaining columns (lat, lon, bird, lat_zone) are generated in
         RANDOM ORDER, matching GReaT's training distribution.

    Returns:
        (lats, lons) — only rows where both values were successfully parsed.
    """
    n_request = int(n_samples * OVERSAMPLE_FACTOR) + 10   # over-sample

    log.info(
        f"  Calling great.sample(n={n_request}, start_col='state_code', "
        f"start_col_dist={{'{state_val}': 1.0}}, T={temperature}) ..."
    )

    df = great_model.sample(
        n_samples      = n_request,
        start_col      = "state_code",
        start_col_dist = {state_val: 1.0},
        temperature    = temperature,
        max_length     = max_length,
        drop_nan       = False,
        device         = device,
    )

    log.info(f"  Generated DataFrame: {df.shape}  columns: {list(df.columns)}")

    # Filter rows where state_code actually matches the target
    if "state_code" in df.columns:
        before = len(df)
        df = df[df["state_code"].astype(str).str.strip() == state_val]
        log.info(f"  After state_code filter: {len(df)}/{before} rows kept")

    lats: List[float] = []
    lons: List[float] = []

    for _, row in df.iterrows():
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                lats.append(lat)
                lons.append(lon)
        except (ValueError, KeyError, TypeError):
            continue
        if len(lats) >= n_samples:
            break

    log.info(f"  Parsed {len(lats)} valid (lat, lon) pairs")
    return lats, lons


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 – Validity checking
# ══════════════════════════════════════════════════════════════════════════════

def check_validity(lats, lons, geom) -> np.ndarray:
    valid = np.zeros(len(lats), dtype=bool)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        pt       = Point(lon, lat)   # shapely: (x=lon, y=lat)
        valid[i] = geom.contains(pt) or geom.touches(pt)
    return valid


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 – Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _draw_state_boundary(ax, geom,
                          fc="#c8e6c9", ec="#1b5e20", lw=2.0, alpha=0.45):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(list(x), list(y), fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=1)
        ax.plot(list(x), list(y), color=ec, lw=lw, zorder=2)
        for interior in poly.interiors:
            xi, yi = interior.xy
            ax.fill(list(xi), list(yi),
                    fc="white", ec=ec, lw=0.8, alpha=1.0, zorder=3)


def plot_scatter_figure(
    results:    Dict[str, Tuple[List[float], List[float], np.ndarray]],
    geometries: Dict[str, object],
    save_path:  Path,
) -> None:
    matplotlib.rcParams.update({
        "font.family":    "DejaVu Sans",
        "font.size":      10,
        "axes.titlesize": 11,
    })

    n_states = len(results)
    ncols    = min(n_states, 2)
    nrows    = (n_states + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes_flat = np.array(axes).flatten()

    fig.suptitle(
        "GReaT Generated Coordinates vs. True State Boundaries  [Method A: Native API]\n"
        "FD Violation: state_code  \u2192  (lat, lon)  |  "
        "Remaining columns generated in random order (GReaT training distribution)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for idx, (abbrev, (lats, lons, valid)) in enumerate(results.items()):
        ax         = axes_flat[idx]
        geom       = geometries[abbrev]
        state_name = ABBREV_TO_NAME.get(abbrev, abbrev)
        n_total    = len(lats)
        n_invalid  = int((~valid).sum())
        n_valid    = int(valid.sum())
        pct_inv    = 100.0 * n_invalid / n_total if n_total > 0 else 0.0

        _draw_state_boundary(ax, geom)

        lats_arr = np.array(lats)
        lons_arr = np.array(lons)

        if n_invalid > 0:
            ax.scatter(
                lons_arr[~valid], lats_arr[~valid],
                c="crimson", marker="x", s=65, linewidths=1.8, alpha=0.75,
                label=f"Invalid  (n = {n_invalid})", zorder=5,
            )
        if n_valid > 0:
            ax.scatter(
                lons_arr[valid], lats_arr[valid],
                c="steelblue", marker="o", s=24, alpha=0.65,
                label=f"Valid  (n = {n_valid})", zorder=4,
            )

        title_color = (
            "#b71c1c" if pct_inv >= 60 else
            "#e65100" if pct_inv >= 30 else
            "#1b5e20"
        )
        ax.set_title(
            f"{state_name} ({abbrev})\n"
            f"Invalid: {n_invalid} / {n_total}  =  {pct_inv:.1f}%",
            fontsize=11, fontweight="bold", color=title_color, pad=8,
        )
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.legend(fontsize=8, loc="best", framealpha=0.85,
                  markerscale=1.2, handlelength=1.5)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=8)

        bds    = geom.bounds
        span   = max(bds[2] - bds[0], bds[3] - bds[1])
        margin = max(span * 0.5, 1.0)
        cap    = span * 3.0

        xlo = bds[0] - margin
        xhi = bds[2] + margin
        ylo = bds[1] - margin
        yhi = bds[3] + margin

        if n_total > 0:
            xlo = max(min(xlo, lons_arr.min() - 0.5), bds[0] - cap)
            xhi = min(max(xhi, lons_arr.max() + 0.5), bds[2] + cap)
            ylo = max(min(ylo, lats_arr.min() - 0.5), bds[1] - cap)
            yhi = min(max(yhi, lats_arr.max() + 0.5), bds[3] + cap)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    for i in range(n_states, len(axes_flat)):
        axes_flat[i].set_visible(False)

    invalid_h = Line2D([], [], color="crimson", marker="x", markersize=9,
                       linestyle="None", label="Invalid  (outside true boundary)")
    valid_h   = Line2D([], [], color="steelblue", marker="o", markersize=8,
                       linestyle="None", label="Valid  (inside true boundary)")
    state_h   = mpatches.Patch(facecolor="#c8e6c9", edgecolor="#1b5e20", alpha=0.9,
                                label="True state boundary")
    fig.legend(
        handles=[invalid_h, valid_h, state_h],
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=3, fontsize=9, frameon=True,
    )

    annotation = (
        "Method A: GReaT native sampling — state_code is fixed as start column, "
        "remaining columns (lat, lon, bird, lat_zone)\n"
        "are generated in random order matching the training distribution. "
        "FD violation = generated (lat, lon) falls outside the true state boundary."
    )
    fig.text(
        0.5, -0.08, annotation,
        ha="center", va="top", fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.0, 1, 1.0])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GReaT Scatter Plot Method A -- native API joint generation"
    )
    parser.add_argument("--model-dir",   type=str,   default=str(MODEL_DIR))
    parser.add_argument("--n-samples",   type=int,   default=N_SAMPLES_PER_STATE,
                        help="Target valid samples per state")
    parser.add_argument("--states",      nargs="+",  default=DEFAULT_STATES, metavar="STATE")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ── Load training data to detect state_code format ────────────────────────
    if not TRAIN_CSV.exists():
        log.error(f"Training CSV not found: {TRAIN_CSV}")
        sys.exit(1)

    train_df   = pd.read_csv(TRAIN_CSV)
    sc_sample  = str(train_df["state_code"].dropna().iloc[0])
    use_abbrev = len(sc_sample) <= 3
    log.info(f"state_code format: {'2-letter abbreviation' if use_abbrev else 'full name'}")
    log.info(f"Sample values: {train_df['state_code'].unique()[:6].tolist()}")

    if use_abbrev:
        available = {str(s).upper() for s in train_df["state_code"].unique()}
    else:
        available = {
            NAME_TO_ABBREV[n]
            for n in train_df["state_code"].astype(str).unique()
            if n in NAME_TO_ABBREV
        }

    target_states = [s.upper() for s in args.states]
    target_states = [s for s in target_states if s in available]
    if not target_states:
        log.error(f"None of {args.states} found in training data.")
        sys.exit(1)
    log.info(f"Target states: {target_states}")

    # ── Load boundaries ───────────────────────────────────────────────────────
    geometries    = load_state_geometries(target_states)
    target_states = [s for s in target_states if s in geometries]

    # ── Load GReaT model ──────────────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        log.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    log.info(f"Loading GReaT model from {model_dir} ...")
    from be_great import GReaT
    great = GReaT.load_from_dir(str(model_dir))

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device_str}")

    # ── Generate + check validity ─────────────────────────────────────────────
    results: Dict[str, Tuple[List[float], List[float], np.ndarray]] = {}

    for abbrev in target_states:
        log.info(f"\n=== {ABBREV_TO_NAME.get(abbrev, abbrev)} ({abbrev}) ===")

        prompt_val = abbrev if use_abbrev else ABBREV_TO_NAME.get(abbrev, abbrev)

        lats, lons = sample_via_great_api(
            great_model = great,
            state_val   = prompt_val,
            n_samples   = args.n_samples,
            device      = device_str,
            temperature = args.temperature,
        )

        if not lats:
            log.warning(f"  No parseable rows for {abbrev}; skipping.")
            continue

        valid = check_validity(lats, lons, geometries[abbrev])
        results[abbrev] = (lats, lons, valid)

        n_inv = int((~valid).sum())
        log.info(
            f"  -> {n_inv}/{len(lats)} invalid  ({100 * n_inv / len(lats):.1f}%)"
        )

    if not results:
        log.error("No valid results for any state.")
        sys.exit(1)

    # ── Save statistics ────────────────────────────────────────────────────────
    rows = []
    for abbrev, (lats, lons, valid) in results.items():
        nt = len(lats)
        ni = int((~valid).sum())
        rows.append({
            "state_code":  abbrev,
            "state_name":  ABBREV_TO_NAME.get(abbrev, abbrev),
            "n_generated": nt,
            "n_valid":     nt - ni,
            "n_invalid":   ni,
            "pct_invalid": round(100.0 * ni / nt, 2) if nt > 0 else 0.0,
        })

    stats_df   = pd.DataFrame(rows)
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    stats_path = results_dir / "scatter_results_A.csv"
    stats_df.to_csv(stats_path, index=False)
    log.info(f"\nStatistics:\n{stats_df.to_string(index=False)}")
    log.info(f"Saved: {stats_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_scatter_figure(
        results    = results,
        geometries = geometries,
        save_path  = results_dir / "scatter_validity_A.png",
    )

    log.info("\n=== Done ===")
    log.info(f"  scatter_validity_A.png  -- Figure 1b (Method A)")
    log.info(f"  scatter_results_A.csv   -- per-state statistics")


if __name__ == "__main__":
    main()
