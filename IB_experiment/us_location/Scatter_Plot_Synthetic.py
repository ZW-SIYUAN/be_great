"""
IB_experiment/us_location/Scatter_Plot_Synthetic.py
=====================================================
Figure: GReaT Coordinate Validity -- Synthetic Data Visualization

Loads the pre-generated synthetic CSV (us_location_synthetic.csv) and
visualises each state's (lat, lon) points against true state boundaries,
colouring points as valid (inside boundary) or invalid (outside boundary).

No model loading required -- reads the synthetic data directly.

Usage:
  python Scatter_Plot_Synthetic.py                        # top-4 states by count
  python Scatter_Plot_Synthetic.py --states OK SD MA ME   # custom state list
  python Scatter_Plot_Synthetic.py --states ALL           # every state in CSV

Outputs (IB_experiment/us_location/results/):
  scatter_validity_synthetic.png   -- panel figure
  scatter_results_synthetic.csv    -- per-state validity statistics
"""

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
HERE        = Path(__file__).parent
DATA_DIR    = HERE / "data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SYNTH_CSV   = DATA_DIR / "us_location_synthetic.csv"

_GEOJSON_URL   = (
    "https://raw.githubusercontent.com/python-visualization/"
    "folium/master/examples/data/us-states.json"
)
_GEOJSON_CACHE = HERE / "us_states_boundaries.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 – State boundary loading
# ══════════════════════════════════════════════════════════════════════════════

def load_state_geometries(target_abbrevs: List[str]) -> Dict[str, object]:
    if not _GEOJSON_CACHE.exists():
        log.info("Downloading US state boundaries ...")
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
# Part 2 – Load synthetic data
# ══════════════════════════════════════════════════════════════════════════════

def load_synthetic_data(
    csv_path: Path,
    target_states: List[str],
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Read the synthetic CSV and return {abbrev: (lats, lons)} per state."""
    df = pd.read_csv(csv_path)
    log.info(f"Loaded synthetic CSV: {df.shape}  columns: {list(df.columns)}")

    # Normalise state_code column
    if "state_code" not in df.columns:
        log.error("Column 'state_code' not found in synthetic CSV.")
        sys.exit(1)

    df["state_code"] = df["state_code"].astype(str).str.strip().str.upper()

    results: Dict[str, Tuple[List[float], List[float]]] = {}
    for abbrev in target_states:
        sub = df[df["state_code"] == abbrev].copy()
        lats, lons = [], []
        for _, row in sub.iterrows():
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    lats.append(lat)
                    lons.append(lon)
            except (ValueError, KeyError, TypeError):
                continue
        log.info(f"  {abbrev}: {len(lats)} valid coordinate rows")
        if lats:
            results[abbrev] = (lats, lons)
        else:
            log.warning(f"  No parseable (lat, lon) rows for {abbrev}; skipping.")
    return results


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
    n_cols:     int = 2,
) -> None:
    matplotlib.rcParams.update({
        "font.family":    "DejaVu Sans",
        "font.size":      10,
        "axes.titlesize": 11,
    })

    n_states = len(results)
    ncols    = min(n_states, n_cols)
    nrows    = (n_states + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes_flat = np.array(axes).flatten() if n_states > 1 else [axes]

    fig.suptitle(
        "GReaT Generated Coordinates vs. True State Boundaries  [Synthetic Data]\n"
        "FD Violation check: state_code  \u2192  (lat, lon)  |  "
        "Pre-generated synthetic samples from us_location_synthetic.csv",
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
        "Synthetic data loaded from us_location_synthetic.csv. "
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
        description="GReaT Scatter Plot -- Synthetic Data Visualization"
    )
    parser.add_argument(
        "--states", nargs="+", default=None, metavar="STATE",
        help='State abbreviations to plot, or "ALL" for every state in the CSV. '
             'Defaults to top-4 states by sample count.',
    )
    parser.add_argument(
        "--n-cols", type=int, default=2,
        help="Number of columns in the figure grid (default: 2)",
    )
    parser.add_argument(
        "--synth-csv", type=str, default=str(SYNTH_CSV),
        help="Path to the synthetic CSV file",
    )
    args = parser.parse_args()

    synth_path = Path(args.synth_csv)
    if not synth_path.exists():
        log.error(f"Synthetic CSV not found: {synth_path}")
        sys.exit(1)

    # ── Determine target states ───────────────────────────────────────────────
    df_all = pd.read_csv(synth_path)
    df_all["state_code"] = df_all["state_code"].astype(str).str.strip().str.upper()
    counts = df_all["state_code"].value_counts()
    available_states = list(counts.index)

    if args.states is None:
        # Default: top-4 states by sample count
        target_states = available_states[:4]
        log.info(f"No --states given; using top-4 by count: {target_states}")
    elif len(args.states) == 1 and args.states[0].upper() == "ALL":
        target_states = available_states
        log.info(f"Using all {len(target_states)} states in CSV")
    else:
        target_states = [s.upper() for s in args.states]
        missing = [s for s in target_states if s not in available_states]
        if missing:
            log.warning(f"States not in synthetic CSV: {missing}")
        target_states = [s for s in target_states if s in available_states]

    if not target_states:
        log.error("No valid target states found.")
        sys.exit(1)

    log.info(f"Target states: {target_states}")
    for s in target_states:
        log.info(f"  {s}: {counts.get(s, 0)} synthetic samples")

    # ── Load boundaries ───────────────────────────────────────────────────────
    geometries    = load_state_geometries(target_states)
    target_states = [s for s in target_states if s in geometries]
    if not target_states:
        log.error("No boundary data found for any target state.")
        sys.exit(1)

    # ── Load synthetic data ───────────────────────────────────────────────────
    data = load_synthetic_data(synth_path, target_states)
    if not data:
        log.error("No parseable data for any state.")
        sys.exit(1)

    # ── Check validity ────────────────────────────────────────────────────────
    results: Dict[str, Tuple[List[float], List[float], np.ndarray]] = {}
    for abbrev, (lats, lons) in data.items():
        valid = check_validity(lats, lons, geometries[abbrev])
        results[abbrev] = (lats, lons, valid)
        n_inv = int((~valid).sum())
        log.info(
            f"  {abbrev}: {n_inv}/{len(lats)} invalid  "
            f"({100 * n_inv / len(lats):.1f}%)"
        )

    # ── Save statistics ───────────────────────────────────────────────────────
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
    stats_path = RESULTS_DIR / "scatter_results_synthetic.csv"
    stats_df.to_csv(stats_path, index=False)
    log.info(f"\nStatistics:\n{stats_df.to_string(index=False)}")
    log.info(f"Saved: {stats_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig_path = RESULTS_DIR / "scatter_validity_synthetic.png"
    plot_scatter_figure(
        results    = results,
        geometries = geometries,
        save_path  = fig_path,
        n_cols     = args.n_cols,
    )

    log.info("\n=== Done ===")
    log.info(f"  {fig_path.name}   -- scatter plot figure")
    log.info(f"  {stats_path.name} -- per-state statistics")


if __name__ == "__main__":
    main()
