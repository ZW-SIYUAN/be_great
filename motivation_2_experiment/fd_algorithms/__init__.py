"""
fd_algorithms
=============
Pure-Python implementations of two exact Functional Dependency discovery
algorithms: TANE and HyFD.

Public API
----------
  from fd_algorithms import tane, hyfd

  fds, stats = tane.discover(df)
  fds, stats = hyfd.discover(df, n_sample_pairs=60)

Each returns:
  fds   : list[tuple[frozenset[str], str]]   – minimal (LHS, RHS) pairs
  stats : dict                                – internal counters for analysis

See the individual module docstrings for full algorithm descriptions.
"""

from . import hyfd, tane
from .partition import StrippedPartition

__all__ = ["tane", "hyfd", "StrippedPartition"]
