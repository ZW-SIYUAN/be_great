"""
fd_algorithms/tane.py
=====================
TANE – an exact, level-wise algorithm for Functional Dependency discovery.

Reference
---------
  Huhtala, Y., Kärkkäinen, J., Penfold, P., & Toivonen, H. (1999).
  TANE: An Efficient Algorithm for Discovering Functional and Approximate
  Dependencies. The Computer Journal, 42(2), 100–111.

Algorithm outline
-----------------
TANE performs a breadth-first traversal of the power-set lattice of
attributes, processing candidates in order of increasing LHS size.

  Level 1 :  single-attribute sets  { {A} : A ∈ R }
  Level k+1:  join pairs from level k that share a (k-1)-element prefix

At each level k, for every candidate X (|X| = k) and every attribute A ∉ X:
  - Check FD X → A using the stripped partition of X.
  - If it holds, record X → A and mark A as "determined by X".

Pruning (key efficiency gain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  C⁺(X):  set of attributes already known to be determined by X.
           Maintained via inheritance from proper subsets of X.
           Avoids re-checking FDs already implied by a smaller LHS.

  Key elimination:  if X determines every attribute, no extension of X can
                    yield a *new* (minimal) FD → X is excluded from joining.

Minimality
~~~~~~~~~~
The level-wise order guarantees that when we first record X → A at level k,
no proper subset of X has yet been shown to determine A.  However, due to
C⁺-inheritance propagation, redundant FDs (non-minimal) can occasionally
slip through, so a final minimality filter is applied.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .partition import StrippedPartition


# ── Lattice utilities ─────────────────────────────────────────────────────────

def _prefix_block_join(
    attr_sets: list[frozenset[str]],
) -> list[frozenset[str]]:
    """
    Generate the next-level candidates using the prefix-block join.

    Two sets X and Y (|X| = |Y| = k) are joined iff they share the same
    first k-1 attributes in sorted (lexicographic) order.  This ensures
    every (k+1)-element candidate is generated exactly once.

    Parameters
    ----------
    attr_sets : list[frozenset[str]]
        Candidate sets at the current level (all of the same size k).

    Returns
    -------
    list[frozenset[str]]
        New candidate sets of size k+1 (duplicates are removed).
    """
    ordered = sorted(attr_sets, key=lambda s: tuple(sorted(s)))
    seen: set[frozenset] = set()
    results: list[frozenset] = []

    for i, X in enumerate(ordered):
        xs = tuple(sorted(X))
        for Y in ordered[i + 1 :]:
            ys = tuple(sorted(Y))
            if xs[:-1] != ys[:-1]:
                break           # ordered, so no further match possible
            new_set = X | Y
            if new_set not in seen:
                seen.add(new_set)
                results.append(new_set)

    return results


def _filter_minimal(
    fds: list[tuple[frozenset[str], str]],
) -> list[tuple[frozenset[str], str]]:
    """
    Remove non-minimal FDs: keep X → A only if no proper subset Y ⊊ X
    with Y → A was also found.

    Since TANE discovers FDs bottom-up, non-minimal duplicates are rare
    but possible when C⁺-inheritance misses some paths.

    Complexity
    ----------
    O(F² · k) where F = number of raw FDs and k = max LHS size.
    """
    by_rhs: dict[str, list[frozenset]] = defaultdict(list)
    for lhs, rhs in fds:
        by_rhs[rhs].append(lhs)

    minimal: list[tuple[frozenset[str], str]] = []
    for rhs, lhs_list in by_rhs.items():
        for lhs in lhs_list:
            if not any(other < lhs for other in lhs_list):   # no proper subset
                minimal.append((lhs, rhs))

    return minimal


# ── Main algorithm ────────────────────────────────────────────────────────────

def discover(
    df: pd.DataFrame,
) -> tuple[list[tuple[frozenset[str], str]], dict]:
    """
    Run TANE on *df* and return all minimal functional dependencies.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.  All dtypes are supported; string/object columns are
        compared by equality (hash-equivalent).

    Returns
    -------
    fds : list[tuple[frozenset[str], str]]
        Minimal FDs as (LHS_frozenset, RHS_column_name) pairs.
    stats : dict
        Internal counters for benchmarking:
          - n_candidates     : total attribute sets evaluated
          - n_fd_checks      : total SP.fd_holds_with() calls
          - n_sp_products    : total stripped-partition products computed
          - n_keys_pruned    : sets removed because they are (super)keys
          - n_raw_fds        : FDs found before minimality filter
          - n_minimal_fds    : FDs after minimality filter
    """
    columns: list[str] = list(df.columns)
    n_attr: int = len(columns)

    # Pre-compute numpy arrays for O(1) element access
    col_arrays: dict[str, np.ndarray] = {c: df[c].to_numpy() for c in columns}

    # ── Initialise partitions and C⁺ for level 1 ─────────────────────────────
    sp_cache: dict[frozenset, StrippedPartition] = {}
    for col in columns:
        sp_cache[frozenset([col])] = StrippedPartition.from_array(col_arrays[col])

    # C⁺[X] = set of attributes determined by X (including X itself trivially)
    c_plus: dict[frozenset, set[str]] = defaultdict(set)
    for col in columns:
        c_plus[frozenset([col])].add(col)

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    raw_fds: list[tuple[frozenset, str]] = []
    stats: dict = {
        "n_candidates":  0,
        "n_fd_checks":   0,
        "n_sp_products": 0,
        "n_keys_pruned": 0,
        "n_raw_fds":     0,
        "n_minimal_fds": 0,
    }

    # Sets that act as (super)keys – no need to extend them further
    key_sets: set[frozenset] = set()

    # ── Level-wise traversal ──────────────────────────────────────────────────
    current_level: list[frozenset] = [frozenset([col]) for col in columns]

    while current_level:
        stats["n_candidates"] += len(current_level)
        valid_for_join: list[frozenset] = []    # non-key sets, eligible for next level

        for X in current_level:
            sp_x = sp_cache[X]

            # ── FD checks for this candidate ──────────────────────────────────
            for A in columns:
                if A in X:
                    continue          # trivial: A ∈ X → A
                if A in c_plus[X]:
                    continue          # already known: some subset of X determines A

                stats["n_fd_checks"] += 1

                if sp_x.fd_holds_with(col_arrays[A]):
                    raw_fds.append((X, A))
                    c_plus[X].add(A)

            # ── Key detection and pruning ─────────────────────────────────────
            # If X determines every column not in X, extending X cannot produce
            # a *new minimal* FD → exclude from join.
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
        valid_set = set(valid_for_join)   # O(1) membership test

        for new_X in _prefix_block_join(valid_for_join):
            # Apriori-style pruning: all (k)-subsets must be valid candidates
            if not all((new_X - {c}) in valid_set for c in new_X):
                continue

            # Compute SP(new_X) incrementally:
            # find any cached subset SP(new_X \ {col}) and apply product_with(col)
            computed = False
            for col in new_X:
                base_key = new_X - {col}
                if base_key in sp_cache:
                    sp_cache[new_X] = sp_cache[base_key].product_with(col_arrays[col])
                    stats["n_sp_products"] += 1
                    computed = True
                    break

            if not computed:
                continue    # should not happen in a correct lattice traversal

            # Inherit C⁺ from all (k)-subsets (union of their determined sets)
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
