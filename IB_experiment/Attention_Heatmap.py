"""
IB_experiment/Attention_Heatmap.py
===================================
Figure 1a: GReaT Attention Heatmap
-- Revealing the Structural Mismatch Between Dense Attention and Sparse Dependencies

Experimental background:
  The us_location dataset has strict functional dependencies (FD):
    (lat, lon) -> state_code   : lat/lon uniquely determines the US state
    (lat, lon) -> lat_zone     : latitude determines climate zone
    state_code -> bird         : state determines the state bird

  GReaT fine-tunes an LLM with random column permutations and standard
  causal self-attention (fully-connected within visible context).
  This script visualizes GReaT's column-level attention distribution
  on us_location_test.csv after fine-tuning, demonstrating that attention
  is "dense and diffuse" -- the attention budget is spread uniformly across
  all columns rather than concentrated on functionally-dependent sources.

Semantic attention aggregation:
  For each query token q belonging to column Q, we compute the fraction
  of its attention budget directed to each key column K:
      frac(Q->K) = sum_{k in col_K} attn[q,k] / sum_{k in any col} attn[q,k]
  This is averaged over all query tokens in column Q and all test samples,
  giving a row-stochastic semantic attention matrix independent of
  positional effects from random column ordering.

Usage:
  # Train from scratch + extract + plot
  python Attention_Heatmap.py

  # Skip training (requires saved model)
  python Attention_Heatmap.py --skip-training

  # Skip training with custom model directory
  python Attention_Heatmap.py --skip-training --model-dir /path/to/model

Outputs (saved to IB_experiment/us_location/):
  attention_heatmap.png           -- 3-panel main figure (paper Figure 1a)
  attention_heatmap_layerwise.png -- layer-wise evolution (supplementary)
  attention_matrix.csv / .npy     -- raw numerical data
"""

import argparse
import sys
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # headless-safe; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent          # IB_experiment/
ROOT       = HERE.parent                    # be_great/
DATA_DIR   = HERE / "us_location"
MODEL_DIR  = DATA_DIR / "us_great_model"
TRAIN_CSV  = DATA_DIR / "us_location_train.csv"
TEST_CSV   = DATA_DIR / "us_location_test.csv"

sys.path.insert(0, str(ROOT))   # ensure be_great is importable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Training hyperparameters (tuned for us_location_train.csv) ────────────────
# Dataset: 16320 rows x 5 columns, lat/lon 2-decimal floats
# Sequence length: ~30-45 tokens (short), allows large batch size
TRAIN_CFG = dict(
    llm                    = "gpt2-medium",  # 345M baseline, standard for this project
    epochs                 = 15,             # 16320/64 ~ 255 steps/epoch x15 ~ 3825 steps
    batch_size             = 64,             # short sequences (~35-45 tok), no VRAM pressure
    float_precision        = 2,              # lat/lon have 2 decimal places
    bf16                   = True,
    dataloader_num_workers = 0,              # Windows: keep 0
    efficient_finetuning   = "lora",
    lora_config = dict(
        r              = 16,
        lora_alpha     = 32,
        target_modules = ["c_attn", "c_proj"],  # GPT-2 Conv1D layer names
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = "CAUSAL_LM",
    ),
)

# ── Attention extraction parameters ──────────────────────────────────────────
N_TEST_SAMPLES  = 200   # test samples to draw
N_ORDERINGS     = 5     # random permutations per sample (mimics GReaT random order)
FLOAT_PRECISION = 2     # must match training
RANDOM_SEED     = 42

# ── Column definitions & functional dependencies ─────────────────────────────
COLUMNS   = ["state_code", "lat", "lon", "bird", "lat_zone"]
COL_SHORT = ["StateCode", "Lat", "Lon", "Bird", "LatZone"]   # short labels for plots
COL_FULL  = ["State Code", "Latitude", "Longitude", "State Bird", "Lat Zone"]
# Semantic index lookup  col_name -> index in COLUMNS
COL_TO_IDX = {col: i for i, col in enumerate(COLUMNS)}

# True functional dependencies: (key_col_idx, query_col_idx)
# Meaning: key_col DETERMINES query_col
FD_EDGES = [
    (1, 0),   # Lat  -> StateCode
    (2, 0),   # Lon  -> StateCode
    (1, 4),   # Lat  -> LatZone
    (2, 4),   # Lon  -> LatZone
    (0, 3),   # StateCode -> Bird
]

# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Train GReaT model
# ══════════════════════════════════════════════════════════════════════════════

def train_great_model(train_df: pd.DataFrame, save_dir: Path):
    """Fine-tune GReaT on us_location_train.csv and save to save_dir."""
    from be_great import GReaT

    log.info(f"Initializing GReaT ({TRAIN_CFG['llm']})...")
    model = GReaT(
        llm                    = TRAIN_CFG["llm"],
        epochs                 = TRAIN_CFG["epochs"],
        batch_size             = TRAIN_CFG["batch_size"],
        float_precision        = TRAIN_CFG["float_precision"],
        bf16                   = TRAIN_CFG["bf16"],
        dataloader_num_workers = TRAIN_CFG["dataloader_num_workers"],
        efficient_finetuning   = TRAIN_CFG["efficient_finetuning"],
        lora_config            = TRAIN_CFG["lora_config"],
        experiment_dir         = str(save_dir / "trainer_ckpt"),
    )

    log.info(f"Fine-tuning on {len(train_df)} rows x {len(train_df.columns)} cols...")
    model.fit(train_df)

    model.save(str(save_dir))
    log.info(f"Model saved to {save_dir}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Attention weight extraction
# ══════════════════════════════════════════════════════════════════════════════

def row_to_great_text(row: dict, col_order: list, float_precision: int = 2) -> str:
    """Convert a data row to GReaT text format.

    Format: "col1 is val1, col2 is val2, ..., coln is valn"
    Matches GReaTDataset._getitem serialization.
    """
    parts = []
    for col in col_order:
        val = row[col]
        if isinstance(val, (float, np.floating)):
            val_str = f"{val:.{float_precision}f}"
            if "." in val_str:
                val_str = val_str.rstrip("0").rstrip(".")
        else:
            val_str = str(val).strip()
        parts.append(f"{col} is {val_str}")
    return ", ".join(parts)


def build_col_char_spans(text: str, row: dict, col_order: list,
                          float_precision: int = 2) -> dict:
    """Compute character spans for each column in the GReaT text.

    Returns: {col_name: (start_char, end_char)} for each column.
    Used to map token positions back to column identities.
    """
    spans = {}
    pos   = 0
    n     = len(col_order)

    for i, col in enumerate(col_order):
        val = row[col]
        if isinstance(val, (float, np.floating)):
            val_str = f"{val:.{float_precision}f}"
            if "." in val_str:
                val_str = val_str.rstrip("0").rstrip(".")
        else:
            val_str = str(val).strip()

        seg = f"{col} is {val_str}"
        spans[col] = (pos, pos + len(seg))
        pos += len(seg)
        if i < n - 1:
            pos += 2   # ", "

    return spans


def tokens_to_col_indices(offsets: list, col_order: list, col_spans: dict) -> list:
    """Map each token to its SEMANTIC column index (index in global COLUMNS list).

    IMPORTANT: returns COL_TO_IDX[col], NOT the positional index in col_order.
    This ensures that the aggregated attention matrix is keyed by semantic
    column identity rather than by the position in the current permutation.

    Returns: list[int], length = seq_len
      -1  => separator token (comma, space) or unrecognized
      0-4 => semantic index in COLUMNS (state_code=0, lat=1, lon=2, bird=3, lat_zone=4)
    """
    token_cols = []
    for (tok_start, tok_end) in offsets:
        if tok_start == tok_end:     # special/zero-length token
            token_cols.append(-1)
            continue
        assigned = -1
        for col in col_order:
            cs, ce = col_spans[col]
            if tok_start < ce and tok_end > cs:
                assigned = COL_TO_IDX[col]   # <-- SEMANTIC index, not positional
                break
        token_cols.append(assigned)
    return token_cols


def _aggregate_to_col_level(avg_attn: np.ndarray,
                             token_cols_arr: np.ndarray,
                             n_cols: int) -> np.ndarray:
    """Aggregate token-level attention to a semantic column-level matrix.

    For each query token q in column Q:
      1. Sum attention to tokens belonging to each key column K
            row_attn[K] = sum_{k: token_cols[k]==K} avg_attn[q, k]
      2. Normalise by total attention to ALL column tokens (excludes separators)
            frac[K] = row_attn[K] / sum(row_attn)
      3. Accumulate into col_attn[Q] and divide by token count of Q

    Result: col_attn[Q, K] = fraction of Q's attention budget spent on K,
    averaged over all Q-tokens.  Rows are approximately stochastic (sum ~1),
    and the causal-masking effect is handled naturally: when K appears after Q
    in the sequence, avg_attn[q, k]=0 for all k in K, so the fraction is 0.

    Returns: (n_cols, n_cols) float32 ndarray
    """
    col_attn = np.zeros((n_cols, n_cols), dtype=np.float32)
    q_counts = np.zeros(n_cols,           dtype=np.float32)
    seq_len  = avg_attn.shape[0]

    for q_idx in range(seq_len):
        q_col = int(token_cols_arr[q_idx])
        if q_col < 0:
            continue

        # Attention fraction going to each column
        row_attn = np.zeros(n_cols, dtype=np.float32)
        for k_idx in range(seq_len):
            k_col = int(token_cols_arr[k_idx])
            if k_col < 0:
                continue
            row_attn[k_col] += avg_attn[q_idx, k_idx]

        total = row_attn.sum()
        if total > 1e-9:
            row_attn /= total        # convert to fraction (row sums to 1)

        col_attn[q_col] += row_attn
        q_counts[q_col] += 1.0

    for i in range(n_cols):
        if q_counts[i] > 0:
            col_attn[i] /= q_counts[i]

    return col_attn


def extract_attention_one_sequence(
    model, tokenizer, text: str, row: dict, col_order: list,
    device: torch.device, n_cols: int = 5, float_precision: int = 2,
) -> np.ndarray:
    """Extract semantic column-level attention for a single GReaT sequence.

    Returns:
        (n_cols, n_cols) float32 ndarray where entry [Q, K] is the fraction
        of Q's attention budget directed to K, averaged over all Q-tokens
        and all transformer layers/heads.
        Returns None on failure.
    """
    # Step 1: tokenize in two calls to avoid transformers version incompatibility
    # (return_tensors="pt" + return_offsets_mapping=True raises in some 4.x versions)
    enc_plain = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets   = enc_plain["offset_mapping"]   # list of (start, end) tuples

    ids_list  = enc_plain["input_ids"]
    if not ids_list:
        return None
    input_ids = torch.tensor([ids_list], dtype=torch.long).to(device)
    attn_mask = torch.ones(1, len(ids_list),  dtype=torch.long).to(device)
    seq_len   = input_ids.shape[1]

    if seq_len < 3:
        return None

    # Step 2: map tokens to semantic column indices
    col_spans      = build_col_char_spans(text, row, col_order, float_precision)
    token_cols     = tokens_to_col_indices(offsets, col_order, col_spans)
    token_cols_arr = np.array(token_cols, dtype=np.int32)

    # Step 3: forward pass with attention output
    with torch.no_grad():
        outputs = model(
            input_ids         = input_ids,
            attention_mask    = attn_mask,
            output_attentions = True,
        )

    # Filter None entries (safety net; should not occur after eager-model switch)
    valid_attentions = [a for a in outputs.attentions if a is not None]
    if not valid_attentions:
        return None

    # Stack -> (n_layers, n_heads, seq_len, seq_len); average over layers & heads
    stacked  = torch.stack([a.squeeze(0) for a in valid_attentions], dim=0)
    avg_attn = stacked.mean(dim=(0, 1)).cpu().float().numpy()

    # Step 4: aggregate to column level (per-token normalised fractions)
    return _aggregate_to_col_level(avg_attn, token_cols_arr, n_cols)


def extract_layerwise_attention(
    model, tokenizer, text: str, row: dict, col_order: list,
    device: torch.device, n_cols: int = 5, float_precision: int = 2,
) -> np.ndarray:
    """Extract per-layer semantic column-level attention.

    Returns: (n_layers, n_cols, n_cols) float32 ndarray, or None on failure.
    """
    enc_plain = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets   = enc_plain["offset_mapping"]

    ids_list  = enc_plain["input_ids"]
    if not ids_list:
        return None
    input_ids = torch.tensor([ids_list], dtype=torch.long).to(device)
    attn_mask = torch.ones(1, len(ids_list), dtype=torch.long).to(device)
    seq_len   = input_ids.shape[1]

    if seq_len < 3:
        return None

    col_spans      = build_col_char_spans(text, row, col_order, float_precision)
    token_cols     = tokens_to_col_indices(offsets, col_order, col_spans)
    token_cols_arr = np.array(token_cols, dtype=np.int32)

    with torch.no_grad():
        outputs = model(
            input_ids         = input_ids,
            attention_mask    = attn_mask,
            output_attentions = True,
        )

    valid_attentions = [a for a in outputs.attentions if a is not None]
    if not valid_attentions:
        return None

    n_layers  = len(valid_attentions)
    layerwise = np.zeros((n_layers, n_cols, n_cols), dtype=np.float32)

    for layer_idx, layer_attn in enumerate(valid_attentions):
        # Average over heads -> (seq_len, seq_len)
        avg_layer = layer_attn.squeeze(0).mean(dim=0).cpu().float().numpy()
        layerwise[layer_idx] = _aggregate_to_col_level(avg_layer, token_cols_arr, n_cols)

    return layerwise


def extract_attention_from_test(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    device: torch.device,
    n_samples:       int = N_TEST_SAMPLES,
    n_orderings:     int = N_ORDERINGS,
    float_precision: int = FLOAT_PRECISION,
) -> tuple:
    """Extract and aggregate semantic column-level attention from test samples.

    Uses n_orderings random column permutations per sample to simulate GReaT's
    training distribution (random column ordering).

    Returns:
        mean_attn    : (5, 5) float32 ndarray -- averaged semantic attention
        layerwise_avg: (n_layers, 5, 5) float32 ndarray -- per-layer averaged
    """
    rng            = np.random.default_rng(RANDOM_SEED)
    sample_indices = rng.choice(len(test_df),
                                size=min(n_samples, len(test_df)),
                                replace=False)
    sample_df      = test_df.iloc[sample_indices].reset_index(drop=True)

    n_cols         = len(COLUMNS)
    total_attn     = np.zeros((n_cols, n_cols), dtype=np.float64)
    total_count    = 0
    layerwise_total = None
    layerwise_count = 0

    model.eval()
    log.info(f"Extracting attention: {len(sample_df)} samples x {n_orderings} "
             f"permutations = {len(sample_df) * n_orderings} forward passes")

    for sample_idx, (_, row) in enumerate(sample_df.iterrows()):
        row_dict = row.to_dict()

        for ord_idx in range(n_orderings):
            # Random column order (mimics GReaT's training distribution)
            col_order   = COLUMNS.copy()
            rng_shuffle = random.Random(RANDOM_SEED + sample_idx * 100 + ord_idx)
            rng_shuffle.shuffle(col_order)

            text     = row_to_great_text(row_dict, col_order, float_precision)
            col_attn = extract_attention_one_sequence(
                model, tokenizer, text, row_dict, col_order,
                device, n_cols, float_precision,
            )
            if col_attn is not None:
                total_attn  += col_attn.astype(np.float64)
                total_count += 1

            # Layer-wise extraction: only for first permutation (memory saving)
            if ord_idx == 0:
                lw = extract_layerwise_attention(
                    model, tokenizer, text, row_dict, col_order,
                    device, n_cols, float_precision,
                )
                if lw is not None:
                    if layerwise_total is None:
                        layerwise_total = lw.astype(np.float64)
                    else:
                        layerwise_total += lw.astype(np.float64)
                    layerwise_count += 1

        if (sample_idx + 1) % 20 == 0:
            log.info(f"  Processed {sample_idx + 1}/{len(sample_df)} samples "
                     f"(valid: {total_count})")

    if total_count == 0:
        raise RuntimeError(
            "No valid attention weights extracted. "
            "Check model and tokenizer configuration."
        )

    mean_attn     = (total_attn / total_count).astype(np.float32)
    layerwise_avg = None
    if layerwise_total is not None and layerwise_count > 0:
        layerwise_avg = (layerwise_total / layerwise_count).astype(np.float32)

    log.info(f"Extraction complete: {total_count} valid sequences aggregated.")
    return mean_attn, layerwise_avg


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _draw_fd_boxes(ax):
    """Overlay red dashed boxes on heatmap cells that correspond to true FDs.

    FD_EDGES format: (key_col_idx, query_col_idx)
    Seaborn heatmap coords: x = key (column axis), y = query (row axis)
    """
    for (key_col, query_col) in FD_EDGES:
        rect = mpatches.FancyBboxPatch(
            (key_col, query_col), 1, 1,
            boxstyle  = "square,pad=0",
            edgecolor = "red",
            facecolor = "none",
            linewidth = 2.2,
            linestyle = "--",
            transform = ax.transData,
            clip_on   = True,
            zorder    = 5,
        )
        ax.add_patch(rect)


def _build_ideal_fd_matrix(n_cols: int) -> np.ndarray:
    """Construct the ideal sparse attention matrix expected under FD structure.

    Entry [Q, K] = 1 if (K -> Q) is a true functional dependency, else 0.
    For self-attention (Q==K) we set it to 1 (model always attends to itself).
    Then row-normalise so each row sums to 1.
    """
    ideal = np.zeros((n_cols, n_cols), dtype=np.float32)
    for i in range(n_cols):
        ideal[i, i] = 1.0          # self-attention
    for (key_col, query_col) in FD_EDGES:
        ideal[query_col, key_col] = 1.0
    # Row-normalise
    row_sums = ideal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return ideal / row_sums


def plot_attention_heatmap(
    mean_attn: np.ndarray,
    save_path: Path,
    title: str = "GReaT Attention Heatmap -- Structural Mismatch (us_location)",
):
    """Generate the 3-panel main figure (paper Figure 1a).

    Panel (a): Semantic attention fractions (fraction of Q's budget to K)
    Panel (b): Row-normalised attention (percentage per query column)
    Panel (c): GReaT vs Ideal FD -- deviation from ideal sparse attention

    Red dashed boxes mark cells corresponding to true functional dependencies.
    """
    col_labels = COL_SHORT

    # Row-normalised version (already approximately stochastic from aggregation,
    # but re-normalise explicitly for clean display)
    row_sums  = mean_attn.sum(axis=1, keepdims=True)
    row_sums  = np.where(row_sums == 0, 1.0, row_sums)
    norm_attn = mean_attn / row_sums

    # Deviation from ideal FD matrix
    ideal_attn = _build_ideal_fd_matrix(len(COLUMNS))
    deviation  = norm_attn - ideal_attn  # positive -> GReaT attends more than ideal

    matplotlib.rcParams.update({
        "font.family"   : "DejaVu Sans",
        "font.size"     : 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    # ── Panel (a): Semantic Attention Fractions ────────────────────────────────
    ax_a = axes[0]
    sns.heatmap(
        norm_attn,
        ax          = ax_a,
        cmap        = "YlOrRd",
        annot       = True,
        fmt         = ".3f",
        linewidths  = 0.5,
        linecolor   = "white",
        square      = True,
        xticklabels = col_labels,
        yticklabels = col_labels,
        cbar_kws    = {"shrink": 0.75, "label": "Attention Fraction"},
        vmin        = 0.0,
        vmax        = 1.0,
    )
    ax_a.set_title("(a) Semantic Attention Fractions\n(GReaT fine-tuned on us_location)",
                   fontsize=11, pad=6)
    ax_a.set_xlabel("Key Column  [attended to]", labelpad=5)
    ax_a.set_ylabel("Query Column  [attending]", labelpad=5)
    ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax_a.set_yticklabels(ax_a.get_yticklabels(), rotation=0,  fontsize=9)
    _draw_fd_boxes(ax_a)

    # ── Panel (b): Ideal Sparse FD Attention ──────────────────────────────────
    ax_b = axes[1]
    sns.heatmap(
        ideal_attn,
        ax          = ax_b,
        cmap        = "Greens",
        annot       = True,
        fmt         = ".2f",
        linewidths  = 0.5,
        linecolor   = "white",
        square      = True,
        xticklabels = col_labels,
        yticklabels = col_labels,
        cbar_kws    = {"shrink": 0.75, "label": "Ideal Attention Fraction"},
        vmin        = 0.0,
        vmax        = 1.0,
    )
    ax_b.set_title("(b) Ideal Sparse FD Attention\n(oracle: attend only to FD sources)",
                   fontsize=11, pad=6)
    ax_b.set_xlabel("Key Column  [attended to]", labelpad=5)
    ax_b.set_ylabel("Query Column  [attending]", labelpad=5)
    ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax_b.set_yticklabels(ax_b.get_yticklabels(), rotation=0,  fontsize=9)
    _draw_fd_boxes(ax_b)

    # ── Panel (c): Deviation = GReaT - Ideal ──────────────────────────────────
    ax_c = axes[2]
    dev_max = max(abs(deviation).max(), 0.01)
    sns.heatmap(
        deviation,
        ax          = ax_c,
        cmap        = "RdBu_r",
        annot       = True,
        fmt         = "+.3f",
        linewidths  = 0.5,
        linecolor   = "white",
        square      = True,
        xticklabels = col_labels,
        yticklabels = col_labels,
        cbar_kws    = {"shrink": 0.75, "label": "Deviation (GReaT - Ideal)"},
        center      = 0.0,
        vmin        = -dev_max,
        vmax        = dev_max,
    )
    ax_c.set_title("(c) Deviation: GReaT minus Ideal\n(red=over-attend, blue=under-attend vs FD)",
                   fontsize=11, pad=6)
    ax_c.set_xlabel("Key Column  [attended to]", labelpad=5)
    ax_c.set_ylabel("Query Column  [attending]", labelpad=5)
    ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax_c.set_yticklabels(ax_c.get_yticklabels(), rotation=0,  fontsize=9)
    _draw_fd_boxes(ax_c)

    # ── Legend & Annotation ────────────────────────────────────────────────────
    fd_patch = mpatches.Patch(
        edgecolor="red", facecolor="none", linewidth=2.0, linestyle="--",
        label="True Functional Dependency (FD) cell"
    )
    fig.legend(
        handles=[fd_patch], loc="lower center",
        bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=10, frameon=True,
    )

    annotation = (
        "Ground-truth FDs: (Lat, Lon) -> StateCode,  (Lat, Lon) -> LatZone,  StateCode -> Bird\n"
        "Ideal model: each query column concentrates its full attention budget on its FD source(s).\n"
        "GReaT (observed): attention is near-uniform across all columns -- "
        "no FD-aware selective pattern detected."
    )
    fig.text(
        0.5, -0.11, annotation,
        ha="center", va="top", fontsize=9, color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.0, 1, 1.0])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Main heatmap saved: {save_path}")


def plot_layerwise_heatmap(layerwise_avg: np.ndarray, save_path: Path):
    """Plot layer-wise semantic attention evolution (supplementary figure).

    Shows 6 representative layers from GPT-2-medium's 24 layers, illustrating
    how attention patterns evolve from early (syntactic/positional) to
    late (semantic) layers.
    """
    from matplotlib.gridspec import GridSpec

    n_layers   = layerwise_avg.shape[0]
    col_labels = COL_SHORT

    # Select 6 representative layers (early, middle, late)
    if n_layers <= 6:
        show_layers = list(range(n_layers))
    else:
        show_layers = [int(round(i * (n_layers - 1) / 5)) for i in range(6)]

    # GridSpec: 2 rows x 4 cols -- 3 heatmap cols + 1 narrow colorbar col
    fig = plt.figure(figsize=(17, 10))
    gs  = GridSpec(2, 4, figure=fig,
                   width_ratios=[1, 1, 1, 0.055],
                   wspace=0.38, hspace=0.48)

    heatmap_axes = [fig.add_subplot(gs[r, c])
                    for r in range(2) for c in range(3)]
    cbar_ax = fig.add_subplot(gs[:, 3])   # spans both rows, dedicated colorbar

    fig.suptitle(
        "GReaT Layer-wise Semantic Column Attention  (us_location / gpt2-medium)\n"
        "Row = Query Column, Col = Key Column, Value = Attention Fraction",
        fontsize=13, fontweight="bold",
    )

    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    for plot_idx, layer_idx in enumerate(show_layers):
        ax   = heatmap_axes[plot_idx]
        data = layerwise_avg[layer_idx]

        # Row-normalise
        rs   = data.sum(axis=1, keepdims=True)
        rs   = np.where(rs == 0, 1.0, rs)
        data = data / rs

        sns.heatmap(
            data,
            ax          = ax,
            cmap        = "YlOrRd",
            annot       = True,
            fmt         = ".2f",
            linewidths  = 0.4,
            linecolor   = "white",
            square      = True,
            xticklabels = col_labels,
            yticklabels = col_labels,
            cbar        = False,
            vmin        = 0.0,
            vmax        = 1.0,
        )
        depth_label = (
            "Early" if layer_idx < n_layers // 3
            else ("Mid" if layer_idx < 2 * n_layers // 3 else "Late")
        )
        ax.set_title(f"Layer {layer_idx + 1}  [{depth_label}]", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
        ax.set_xlabel("Key", fontsize=8)
        ax.set_ylabel("Query", fontsize=8)
        _draw_fd_boxes(ax)

    # Hide unused subplots (if fewer than 6 layers)
    for i in range(len(show_layers), len(heatmap_axes)):
        heatmap_axes[i].set_visible(False)

    # Shared colorbar in its dedicated column, spanning both rows
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Row-Normalised Attention Fraction", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Layer-wise heatmap saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GReaT Attention Heatmap Experiment -- Figure 1a"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and load a saved model (requires --model-dir to exist)"
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR),
        help=f"Model directory (default: {MODEL_DIR})"
    )
    parser.add_argument(
        "--n-samples", type=int, default=N_TEST_SAMPLES,
        help="Number of test samples for attention extraction"
    )
    parser.add_argument(
        "--n-orderings", type=int, default=N_ORDERINGS,
        help="Number of random column permutations per sample"
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ── Load data ──────────────────────────────────────────────────────────────
    for path in [TRAIN_CSV, TEST_CSV]:
        if not path.exists():
            log.error(f"Data file not found: {path}")
            sys.exit(1)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    log.info(f"Train: {train_df.shape}  |  Test: {test_df.shape}")

    model_dir = Path(args.model_dir)

    # ── Train or load model ────────────────────────────────────────────────────
    if args.skip_training:
        log.info(f"Loading model from {model_dir} ...")
        if not model_dir.exists():
            log.error(f"Model directory not found: {model_dir}")
            sys.exit(1)
        from be_great import GReaT
        great_model = GReaT.load_from_dir(str(model_dir))
        raw_model   = great_model.model
        tokenizer   = great_model.tokenizer
        log.info("Model loaded.")
    else:
        log.info("=== Training GReaT ===")
        model_dir.mkdir(parents=True, exist_ok=True)
        great_model = train_great_model(train_df, model_dir)
        raw_model   = great_model.model
        tokenizer   = great_model.tokenizer

    # ── Merge LoRA + switch to eager attention ─────────────────────────────────
    # Two issues to resolve simultaneously:
    # 1. PEFT LoRA-wrapped models may return None for some attention layers.
    #    Fix: merge LoRA delta-weights into the base model.
    # 2. PyTorch 2.x + transformers >= 4.35 defaults to GPT2SdpaAttention,
    #    whose forward() does NOT implement output_attentions -- always returns None.
    #    Fix: reload the model with attn_implementation="eager" (GPT2Attention class)
    #         and load the merged state dict.
    from transformers import AutoModelForCausalLM

    # Step 1: merge LoRA if present
    try:
        from peft import PeftModel
        if isinstance(raw_model, PeftModel):
            log.info("Merging LoRA weights into base model ...")
            raw_model = raw_model.merge_and_unload()
            log.info("LoRA merge complete.")
    except ImportError:
        pass

    # Step 2: reload as eager-attention model and transfer merged weights
    log.info(f"Reloading {TRAIN_CFG['llm']} with attn_implementation='eager' ...")
    eager_model = AutoModelForCausalLM.from_pretrained(
        TRAIN_CFG["llm"],
        attn_implementation="eager",
    )
    eager_model.load_state_dict(raw_model.state_dict())
    raw_model = eager_model
    log.info("Switched to GPT2Attention (eager) -- output_attentions will work correctly.")

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    raw_model.to(device)
    raw_model.eval()

    # ── Extract attention ──────────────────────────────────────────────────────
    log.info("=== Extracting attention weights ===")
    mean_attn, layerwise_avg = extract_attention_from_test(
        model           = raw_model,
        tokenizer       = tokenizer,
        test_df         = test_df,
        device          = device,
        n_samples       = args.n_samples,
        n_orderings     = args.n_orderings,
        float_precision = FLOAT_PRECISION,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    row_sums  = mean_attn.sum(axis=1, keepdims=True)
    row_sums  = np.where(row_sums == 0, 1.0, row_sums)
    norm_attn = mean_attn / row_sums

    df_norm = pd.DataFrame(norm_attn, index=COL_SHORT, columns=COL_SHORT)
    log.info("\nSemantic attention matrix (row-normalised, fraction of attention budget):")
    log.info(f"\n{df_norm.round(4).to_string()}")

    ideal   = _build_ideal_fd_matrix(len(COLUMNS))
    df_dev  = pd.DataFrame(norm_attn - ideal, index=COL_SHORT, columns=COL_SHORT)
    log.info("\nDeviation from ideal FD attention (GReaT - Ideal, positive = over-attends):")
    log.info(f"\n{df_dev.round(4).to_string()}")

    # ── Save data ──────────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "attention_matrix.npy", mean_attn)
    df_norm.to_csv(DATA_DIR / "attention_matrix.csv")
    if layerwise_avg is not None:
        np.save(DATA_DIR / "attention_layerwise.npy", layerwise_avg)
    log.info(f"Attention data saved to {DATA_DIR}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    log.info("=== Generating visualisations ===")
    plot_attention_heatmap(
        mean_attn = mean_attn,
        save_path = DATA_DIR / "attention_heatmap.png",
    )

    if layerwise_avg is not None:
        plot_layerwise_heatmap(
            layerwise_avg = layerwise_avg,
            save_path     = DATA_DIR / "attention_heatmap_layerwise.png",
        )

    log.info("=== Done ===")
    log.info(f"Outputs saved to: {DATA_DIR}")
    log.info("  attention_heatmap.png            -- Figure 1a (3-panel main figure)")
    log.info("  attention_heatmap_layerwise.png  -- Layer-wise evolution (supplementary)")
    log.info("  attention_matrix.npy / .csv      -- Raw numerical data")


if __name__ == "__main__":
    main()
