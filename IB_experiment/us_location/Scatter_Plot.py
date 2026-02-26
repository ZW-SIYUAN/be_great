"""
IB_experiment/us_location/Scatter_Plot.py
==========================================
Figure 1b: GReaT Coordinate Validity Scatter Plot
-- FD violation: state_code -> (lat, lon) is NOT enforced during generation.

After fine-tuning GReaT on us_location, we sample rows conditioned on a
fixed state_code value.  An ideal model obeying the FD
    state_code -> (lat, lon)
should produce coordinates inside the given state's true geographic boundary.
GReaT produces many points outside, providing visual evidence of the structural
mismatch diagnosed by the attention heatmap (Figure 1a).

Usage:
  python Scatter_Plot.py                        # default 4 states, load existing model
  python Scatter_Plot.py --states DE FL WA NV   # custom state list
  python Scatter_Plot.py --n-samples 500        # more samples per state

Outputs (IB_experiment/us_location/):
  scatter_validity.png   -- 2x2 panel figure (paper Figure 1b)
  scatter_results.csv    -- per-state validity statistics
"""

import argparse
import json
import logging
import random
import re
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
HERE      = Path(__file__).parent          # IB_experiment/us_location/
ROOT      = HERE.parent.parent             # be_great/
DATA_DIR  = HERE
MODEL_DIR = DATA_DIR / "us_great_model"
TRAIN_CSV = DATA_DIR / "us_location_train.csv"

sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_SAMPLES_PER_STATE = 500
TEMPERATURE         = 0.7
MAX_NEW_TOKENS      = 120
RANDOM_SEED         = 42
# DE=small compact, RI=tiny, FL=distinctive peninsula, NV=inland desert
DEFAULT_STATES      = ["DE", "RI", "FL", "NV"]

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

# GeoJSON source (folium example data, reliably hosted on GitHub)
_GEOJSON_URL   = (
    "https://raw.githubusercontent.com/python-visualization/"
    "folium/master/examples/data/us-states.json"
)
_GEOJSON_CACHE = DATA_DIR / "us_states_boundaries.json"


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 – State boundary loading
# ══════════════════════════════════════════════════════════════════════════════

def load_state_geometries(target_abbrevs: List[str]) -> Dict[str, object]:
    """Download (once, cached locally) and parse US state boundary polygons.

    Returns {state_abbrev: shapely_geometry}.
    """
    if not _GEOJSON_CACHE.exists():
        log.info(f"Downloading US state boundaries from:\n  {_GEOJSON_URL}")
        try:
            urllib.request.urlretrieve(_GEOJSON_URL, _GEOJSON_CACHE)
            log.info(f"Cached to {_GEOJSON_CACHE}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download US state boundaries: {exc}\n"
                f"Manually place us-states.json at {_GEOJSON_CACHE}"
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
        log.warning(f"No boundary data found for: {missing}")

    return geometries


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 – Conditional generation
# ══════════════════════════════════════════════════════════════════════════════

def _parse_lat_lon(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract (lat, lon) from a GReaT-format string.

    GReaT text format: "col is val, col is val, ..."
    Looks for 'lat is <number>' and 'lon is <number>'.
    Returns (None, None) if either value is missing or invalid.
    """
    lat_m = re.search(r'\blat\s+is\s+([-+]?\d+\.?\d*)', text, re.I)
    lon_m = re.search(r'\blon\s+is\s+([-+]?\d+\.?\d*)', text, re.I)
    if not (lat_m and lon_m):
        return None, None
    try:
        lat = float(lat_m.group(1))
        lon = float(lon_m.group(1))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    except ValueError:
        pass
    return None, None


def generate_for_state(
    model,
    tokenizer,
    prompt_val:     str,
    n_samples:      int,
    device:         torch.device,
    temperature:    float = TEMPERATURE,
    max_new_tokens: int   = MAX_NEW_TOKENS,
) -> Tuple[List[float], List[float]]:
    """Generate (lat, lon) pairs conditioned on state_code = prompt_val.

    Constructs the GReaT-format prompt  "state_code is <prompt_val>"
    and lets the model complete the row.

    Returns:
        (lats, lons) – only rows where both values were successfully parsed.
    """
    prompt    = f"state_code is {prompt_val}"
    input_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    lats: List[float] = []
    lons: List[float] = []
    model.eval()
    log.info(f"  Prompt: '{prompt}'  |  generating {n_samples} rows ...")

    for i in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens     = max_new_tokens,
                temperature        = temperature,
                do_sample          = True,
                pad_token_id       = tokenizer.eos_token_id,
                repetition_penalty = 1.1,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        lat, lon = _parse_lat_lon(text)
        if lat is not None:
            lats.append(lat)
            lons.append(lon)

        if (i + 1) % 100 == 0:
            log.info(f"    {i + 1}/{n_samples} generated,  {len(lats)} parsed so far")

    log.info(f"  Parsed {len(lats)}/{n_samples} rows successfully")
    return lats, lons


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 – Validity checking (point-in-polygon)
# ══════════════════════════════════════════════════════════════════════════════

def check_validity(
    lats: List[float],
    lons: List[float],
    geom,
) -> np.ndarray:
    """Return bool array: True = point is inside (or on boundary of) state.

    shapely uses (x, y) = (longitude, latitude) convention.
    """
    valid = np.zeros(len(lats), dtype=bool)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        pt       = Point(lon, lat)
        valid[i] = geom.contains(pt) or geom.touches(pt)
    return valid


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 – Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _draw_state_boundary(ax, geom,
                          fc="#c8e6c9", ec="#1b5e20", lw=2.0, alpha=0.45):
    """Draw a shapely Polygon or MultiPolygon on a matplotlib axes."""
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(list(x), list(y), fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=1)
        ax.plot(list(x), list(y), color=ec, lw=lw, zorder=2)
        # Interior holes (e.g. Great Lakes cutouts in Michigan)
        for interior in poly.interiors:
            xi, yi = interior.xy
            ax.fill(list(xi), list(yi),
                    fc="white", ec=ec, lw=0.8, alpha=1.0, zorder=3)


def plot_scatter_figure(
    results:    Dict[str, Tuple[List[float], List[float], np.ndarray]],
    geometries: Dict[str, object],
    save_path:  Path,
) -> None:
    """Generate Figure 1b: per-state scatter plots (valid vs invalid points).

    Args:
        results:    {abbrev: (lats, lons, valid_mask)}
        geometries: {abbrev: shapely_geometry}
        save_path:  output PNG path
    """
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
        "GReaT Generated Coordinates vs. True State Boundaries\n"
        "FD Violation: state_code  \u2192  (lat, lon)  is NOT enforced during generation",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for idx, (abbrev, (lats, lons, valid)) in enumerate(results.items()):
        ax         = axes_flat[idx]
        geom       = geometries[abbrev]
        state_name = ABBREV_TO_NAME.get(abbrev, abbrev)
        n_total    = len(lats)
        n_invalid  = int((~valid).sum())
        n_valid    = int(valid.sum())
        pct_inv    = 100.0 * n_invalid / n_total if n_total > 0 else 0.0

        # 1. State boundary
        _draw_state_boundary(ax, geom)

        lats_arr = np.array(lats)
        lons_arr = np.array(lons)

        # 2. Invalid points — prominent red X marks
        if n_invalid > 0:
            ax.scatter(
                lons_arr[~valid], lats_arr[~valid],
                c="crimson", marker="x", s=65, linewidths=1.8, alpha=0.75,
                label=f"Invalid  (n = {n_invalid})", zorder=5,
            )

        # 3. Valid points — steel-blue circles
        if n_valid > 0:
            ax.scatter(
                lons_arr[valid], lats_arr[valid],
                c="steelblue", marker="o", s=24, alpha=0.65,
                label=f"Valid  (n = {n_valid})", zorder=4,
            )

        # 4. Colour-coded title
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

        # 5. Axis extent: state bounding box + margin,
        #    expand to show invalid points but cap at 3x state span
        bds    = geom.bounds   # (minlon, minlat, maxlon, maxlat)
        span   = max(bds[2] - bds[0], bds[3] - bds[1])
        margin = max(span * 0.5, 1.0)   # at least 1 degree
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

    # Hide unused subplots
    for i in range(n_states, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Figure-level legend
    invalid_h = Line2D(
        [], [], color="crimson", marker="x", markersize=9,
        linestyle="None", label="Invalid  (outside true boundary)")
    valid_h = Line2D(
        [], [], color="steelblue", marker="o", markersize=8,
        linestyle="None", label="Valid  (inside true boundary)")
    state_h = mpatches.Patch(
        facecolor="#c8e6c9", edgecolor="#1b5e20", alpha=0.9,
        label="True state boundary")
    fig.legend(
        handles=[invalid_h, valid_h, state_h],
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=3, fontsize=9, frameon=True,
    )

    annotation = (
        "Conditional sampling: each row is generated with state_code fixed to the target state.\n"
        "GReaT cannot enforce the geographic FD  state_code \u2192 (lat, lon)  during sampling,\n"
        "producing coordinates that fall anywhere in the learned distribution."
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
        description="GReaT Scatter Plot -- Figure 1b (coordinate validity)"
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR),
        help=f"Path to saved GReaT model (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--n-samples", type=int, default=N_SAMPLES_PER_STATE,
        help="Number of synthetic rows to generate per state",
    )
    parser.add_argument(
        "--states", nargs="+", default=DEFAULT_STATES, metavar="STATE",
        help="2-letter state codes to visualise (default: DE RI FL NV)",
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help="Sampling temperature for generation",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ── Load training data to detect state_code format ────────────────────────
    if not TRAIN_CSV.exists():
        log.error(f"Training CSV not found: {TRAIN_CSV}")
        sys.exit(1)

    train_df  = pd.read_csv(TRAIN_CSV)
    log.info(f"Training data: {train_df.shape}")

    sc_sample  = str(train_df["state_code"].dropna().iloc[0])
    use_abbrev = len(sc_sample) <= 3
    log.info(
        f"state_code format: {'2-letter abbreviation' if use_abbrev else 'full name'}"
    )
    log.info(f"Sample values: {train_df['state_code'].unique()[:6].tolist()}")

    # Build set of abbreviations present in training data
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
        log.error(
            f"None of {args.states} found in training data. "
            f"Available (sample): {sorted(available)[:10]}"
        )
        sys.exit(1)
    log.info(f"Target states: {target_states}")

    # ── Load US state boundary polygons ───────────────────────────────────────
    log.info("Loading state boundary polygons ...")
    geometries    = load_state_geometries(target_states)
    target_states = [s for s in target_states if s in geometries]
    if not target_states:
        log.error("No boundary polygons available for any target state.")
        sys.exit(1)

    # ── Load GReaT model ──────────────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        log.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    log.info(f"Loading GReaT model from {model_dir} ...")
    from be_great import GReaT
    great     = GReaT.load_from_dir(str(model_dir))
    model     = great.model
    tokenizer = great.tokenizer

    # Merge LoRA weights if present (cleaner generation)
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            log.info("Merging LoRA weights ...")
            model = model.merge_and_unload()
    except ImportError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    model.to(device)
    model.eval()

    # ── Conditional generation + validity check ───────────────────────────────
    results: Dict[str, Tuple[List[float], List[float], np.ndarray]] = {}

    for abbrev in target_states:
        log.info(f"\n=== {ABBREV_TO_NAME.get(abbrev, abbrev)} ({abbrev}) ===")

        # Use the value format that matches the training data
        prompt_val = abbrev if use_abbrev else ABBREV_TO_NAME.get(abbrev, abbrev)

        lats, lons = generate_for_state(
            model          = model,
            tokenizer      = tokenizer,
            prompt_val     = prompt_val,
            n_samples      = args.n_samples,
            device         = device,
            temperature    = args.temperature,
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

    # ── Save per-state statistics ──────────────────────────────────────────────
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
    stats_path = DATA_DIR / "scatter_results.csv"
    stats_df.to_csv(stats_path, index=False)
    log.info(f"\nStatistics:\n{stats_df.to_string(index=False)}")
    log.info(f"Saved: {stats_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    log.info("\nGenerating figure ...")
    plot_scatter_figure(
        results    = results,
        geometries = geometries,
        save_path  = DATA_DIR / "scatter_validity.png",
    )

    log.info("\n=== Done ===")
    log.info(f"Outputs saved to: {DATA_DIR}")
    log.info("  scatter_validity.png  -- Figure 1b")
    log.info("  scatter_results.csv   -- per-state statistics")


if __name__ == "__main__":
    main()
