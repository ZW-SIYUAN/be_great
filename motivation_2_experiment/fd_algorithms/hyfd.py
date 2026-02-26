"""
fd_algorithms/hyfd.py
=====================
HyFD – a Hybrid Approach to Functional Dependency Discovery.

Reference
---------
  Papenbrock, T., & Naumann, F. (2016).
  A Hybrid Approach to Functional Dependency Discovery.
  Proceedings of the 2016 ACM SIGMOD International Conference on Management
  of Data (SIGMOD '16), pp. 821–833.

Algorithm outline
-----------------
HyFD combines a fast *sampling phase* (Phase 1) with an exact *PLI-based
validation phase* (Phase 2).

Phase 1 – Random Sampling  (negative-evidence generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~
  Sample N random row pairs (rᵢ, rⱼ) from the dataset.
  For each pair:
    agree_set   = { A : rᵢ[A] = rⱼ[A] }    (columns where rows agree)
    disagree_set = R - agree_set              (columns where rows differ)

  For every A ∈ disagree_set and any X ⊆ agree_set:
    The pair is an FD-violation witness for X → A:
      rᵢ and rⱼ agree on every attribute in X but disagree on A.

  We store these witnesses in a lookup table:
    witnesses[A] = list of agree_sets that witness "something → A is violated"

  A candidate (X, A) is *refuted by sampling* iff
    ∃ S ∈ witnesses[A]  such that  X ⊆ S.
  These candidates are skipped in Phase 2, reducing costly PLI computations.

Phase 2 – PLI-based Validation  (same core as TANE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Level-wise lattice traversal with stripped-partition checks.
  Any (X, A) pair NOT refuted by sampling undergoes exact verification via
  SP(X).fd_holds_with(col_A).

Key differences from TANE
~~~~~~~~~~~~~~~~~~~~~~~~~
  - Sampling pre-filters the search space before PLI validation.
  - In practice, 60–200 sample pairs refute a large fraction of candidates
    on real-world data, yielding significant runtime savings.
  - Both algorithms are *exact* (no false positives or false negatives).

Sampling size guideline
~~~~~~~~~~~~~~~~~~~~~~~
  Default: n_sample_pairs = 3 × n_columns (heuristic from the original paper).
  For our 13-column synthetic dataset this is 39; we default to 60 for safety.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .partition import StrippedPartition
from .tane import _filter_minimal, _prefix_block_join


# ── Sampling utilities ────────────────────────────────────────────────────────

def _build_witnesses(
    col_arrays: dict[str, np.ndarray],
    columns: list[str],
    n_sample_pairs: int,
    rng: np.random.Generator,
) -> dict[str, list[frozenset[str]]]:
    """
    Phase 1: sample row pairs and build the non-FD witness table.

    Returns
    -------
    witnesses : dict[str, list[frozenset[str]]]
        witnesses[A] = list of agree-sets S such that X ⊆ S witnesses X→A
        is violated (rows agree on all of S but disagree on A).
    """
    n_rows = len(next(iter(col_arrays.values())))
    witnesses: dict[str, list[frozenset]] = defaultdict(list)

    # Draw unique row-index pairs
    indices = np.arange(n_rows)
    pair_i = rng.choice(indices, size=n_sample_pairs, replace=True)
    pair_j = rng.choice(indices, size=n_sample_pairs, replace=True)

    for i, j in zip(pair_i, pair_j):
        if i == j:
            continue

        agree_set = frozenset(
            col for col in columns
            if col_arrays[col][i] == col_arrays[col][j]
        )
        disagree_set = set(columns) - agree_set

        # Record one witness per disagreeing column
        for A in disagree_set:
            witnesses[A].append(agree_set)

    return dict(witnesses)


def _is_refuted(
    X: frozenset[str],
    A: str,
    witnesses: dict[str, list[frozenset]],
) -> bool:
    """
    Return True iff sampling has already proven that X → A does NOT hold.

    X → A is refuted iff there exists a witness S ∈ witnesses[A] such that
    X ⊆ S  (the sampled pair agrees on all of X but disagrees on A).

    Parameters
    ----------
    X : frozenset[str]
        Candidate left-hand side.
    A : str
        Candidate right-hand side attribute.
    witnesses : dict
        Precomputed witness table from _build_witnesses().
    """
    for S in witnesses.get(A, []):
        if X <= S:          # X is a subset of the agree-set → FD X→A violated
            return True
    return False


# ── Main algorithm ────────────────────────────────────────────────────────────

def discover(
    df: pd.DataFrame,
    n_sample_pairs: int = 60,
    seed: int = 0,
) -> tuple[list[tuple[frozenset[str], str]], dict]:
    """
    Run HyFD on *df* and return all minimal functional dependencies.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.  All dtypes supported.
    n_sample_pairs : int
        Number of random row pairs to draw in Phase 1.
        Larger values prune more aggressively but increase Phase 1 cost.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    fds : list[tuple[frozenset[str], str]]
        Minimal FDs as (LHS_frozenset, RHS_column_name) pairs.
    stats : dict
        Internal counters for benchmarking:
          - n_sample_pairs        : row pairs sampled in Phase 1
          - n_witnesses_total     : total witness entries across all RHS cols
          - n_refuted_by_sampling : candidate (X,A) pairs skipped via Phase 1
          - n_fd_checks           : PLI fd_holds_with() calls in Phase 2
          - n_sp_products         : stripped-partition products computed
          - n_keys_pruned         : candidates removed as (super)keys
          - n_raw_fds             : FDs found before minimality filter
          - n_minimal_fds         : FDs after minimality filter
    """
    columns: list[str] = list(df.columns)
    col_arrays: dict[str, np.ndarray] = {c: df[c].to_numpy() for c in columns}
    rng = np.random.default_rng(seed)

    # ── Phase 1: Sampling ─────────────────────────────────────────────────────
    witnesses = _build_witnesses(col_arrays, columns, n_sample_pairs, rng)

    stats: dict = {
        "n_sample_pairs":         n_sample_pairs,
        "n_witnesses_total":      sum(len(v) for v in witnesses.values()),
        "n_refuted_by_sampling":  0,
        "n_fd_checks":            0,
        "n_sp_products":          0,
        "n_keys_pruned":          0,
        "n_raw_fds":              0,
        "n_minimal_fds":          0,
    }

    # ── Phase 2: PLI-based validation, guided by Phase 1 ─────────────────────

    # Initialise single-attribute partitions
    sp_cache: dict[frozenset, StrippedPartition] = {
        frozenset([col]): StrippedPartition.from_array(col_arrays[col])
        for col in columns
    }

    # C⁺[X]: attributes already determined by X
    c_plus: dict[frozenset, set[str]] = defaultdict(set)
    for col in columns:
        c_plus[frozenset([col])].add(col)

    raw_fds: list[tuple[frozenset, str]] = []
    key_sets: set[frozenset] = set()
    current_level: list[frozenset] = [frozenset([col]) for col in columns]

    while current_level:
        valid_for_join: list[frozenset] = []

        for X in current_level:
            sp_x = sp_cache[X]

            # ── Check each potential RHS ──────────────────────────────────────
            for A in columns:
                if A in X:
                    continue
                if A in c_plus[X]:
                    continue        # already determined by a subset of X

                # HyFD Phase 1 pre-filter: skip if sampling has already refuted X→A
                if _is_refuted(X, A, witnesses):
                    stats["n_refuted_by_sampling"] += 1
                    continue

                # Phase 2 exact verification
                stats["n_fd_checks"] += 1
                if sp_x.fd_holds_with(col_arrays[A]):
                    raw_fds.append((X, A))
                    c_plus[X].add(A)

            # ── Key detection ─────────────────────────────────────────────────
            undetermined = set(columns) - X - c_plus[X]
            if not undetermined:
                key_sets.add(X)
                stats["n_keys_pruned"] += 1
            else:
                valid_for_join.append(X)

        # ── Generate next level ───────────────────────────────────────────────
        if len(valid_for_join) < 2:
            break

        next_level: list[frozenset] = []
        valid_set = set(valid_for_join)

        for new_X in _prefix_block_join(valid_for_join):
            if not all((new_X - {c}) in valid_set for c in new_X):
                continue

            # Compute SP(new_X) from any cached subset
            computed = False
            for col in new_X:
                base_key = new_X - {col}
                if base_key in sp_cache:
                    sp_cache[new_X] = sp_cache[base_key].product_with(col_arrays[col])
                    stats["n_sp_products"] += 1
                    computed = True
                    break

            if not computed:
                continue

            # Inherit C⁺
            inherited: set[str] = set()
            for col in new_X:
                inherited |= c_plus.get(new_X - {col}, set())
            c_plus[new_X] = inherited

            next_level.append(new_X)

        current_level = next_level

    # ── Minimality filter ─────────────────────────────────────────────────────
    stats["n_raw_fds"] = len(raw_fds)
    minimal = _filter_minimal(raw_fds)
    stats["n_minimal_fds"] = len(minimal)

    return minimal, stats
