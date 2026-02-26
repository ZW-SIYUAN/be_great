"""
fd_algorithms/partition.py
==========================
Stripped Partition (SP) – the core data structure for exact FD discovery.

A partition π(X) groups rows by their combined X-values into equivalence
classes.  The *stripped* variant discards singleton classes (rows with a
unique X-value), because singletons can never witness an FD violation.

Theory
------
  FD X → A holds  ⟺  π(X ∪ {A}) = π(X)
                  ⟺  every class C ∈ π(X) is homogeneous in A
                  ⟺  SP(X).fd_holds_with(col_A)  [implemented below]

  SP(X ∪ {A}) is computed incrementally:
    for each class C ∈ SP(X):
      split C into sub-groups by A-value
      keep sub-groups of size ≥ 2 → new classes of SP(X ∪ {A})

  If no class is split, SP(X ∪ {A}) ≡ SP(X) and FD X→A holds.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StrippedPartition:
    """
    Stripped partition of rows induced by a set of attributes X.

    Attributes
    ----------
    clusters : list[tuple[int, ...]]
        Equivalence classes (sorted row-index tuples) of size ≥ 2.
    n_rows : int
        Total number of rows in the dataset.
    """

    clusters: list[tuple[int, ...]]
    n_rows: int

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_array(cls, values: np.ndarray) -> StrippedPartition:
        """
        Build the single-attribute SP for column values.

        Parameters
        ----------
        values : np.ndarray, shape (n_rows,)
            Column values (any dtype; object arrays for strings are handled).

        Returns
        -------
        StrippedPartition
            Groups of row indices sharing the same value, singletons removed.
        """
        groups: dict = defaultdict(list)
        for idx in range(len(values)):
            groups[values[idx]].append(idx)

        return cls(
            clusters=[tuple(g) for g in groups.values() if len(g) >= 2],
            n_rows=len(values),
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def sum_size(self) -> int:
        """Total number of rows across all equivalence classes."""
        return sum(len(c) for c in self.clusters)

    @property
    def n_clusters(self) -> int:
        """Number of non-singleton equivalence classes."""
        return len(self.clusters)

    # ── Core operations ───────────────────────────────────────────────────────

    def product_with(self, col_values: np.ndarray) -> StrippedPartition:
        """
        Compute the partition product  SP(X) ⊗ SP({A}),  given raw column A.

        Each class C ∈ SP(X) is split by the values of A.  Sub-groups of
        size ≥ 2 become the classes of SP(X ∪ {A}).

        Parameters
        ----------
        col_values : np.ndarray, shape (n_rows,)
            Raw values of the new attribute A.

        Returns
        -------
        StrippedPartition
            SP for X ∪ {A}.

        Complexity
        ----------
        O(sum of cluster sizes) = O(n_rows) in the worst case.
        """
        new_clusters: list[tuple[int, ...]] = []

        for cluster in self.clusters:
            sub_groups: dict = defaultdict(list)
            for row in cluster:
                sub_groups[col_values[row]].append(row)
            for sub in sub_groups.values():
                if len(sub) >= 2:
                    new_clusters.append(tuple(sub))

        return StrippedPartition(clusters=new_clusters, n_rows=self.n_rows)

    def fd_holds_with(self, col_values: np.ndarray) -> bool:
        """
        Return True iff FD (attributes of self) → A holds in the data.

        Checks that every equivalence class in SP(X) is homogeneous under A:
        if every row in a class shares the same A-value, no class is split,
        which is exactly the FD condition.

        This is equivalent to (but faster than) computing the product partition
        and comparing sum_size, because we can exit early on the first violation.

        Parameters
        ----------
        col_values : np.ndarray, shape (n_rows,)
            Raw values of the right-hand side attribute A.

        Returns
        -------
        bool
            True iff the FD holds exactly (zero errors).
        """
        for cluster in self.clusters:
            anchor = col_values[cluster[0]]
            for row in cluster[1:]:
                if col_values[row] != anchor:
                    return False
        return True
