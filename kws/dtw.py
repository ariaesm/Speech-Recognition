"""Dynamic time warping for comparing variable-length feature sequences."""

from __future__ import annotations

import numpy as np


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every frame of ``a`` (n, d) and ``b`` (m, d)."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature dimensions differ: {a.shape[1]} vs {b.shape[1]}")
    sq = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0.0))


def dtw(
    a: np.ndarray, b: np.ndarray, band: int | None = None, return_path: bool = False
) -> float | tuple[float, list[tuple[int, int]]]:
    """DTW distance with steps (1,0), (0,1), (1,1).

    The accumulated cost is divided by ``n + m`` so sequences of different
    lengths are comparable. ``band`` is an optional Sakoe-Chiba radius (in
    frames, widened automatically to cover the length difference).
    """
    cost = pairwise_distances(a, b)
    n, m = cost.shape
    if band is not None:
        band = max(band, abs(n - m))
        ii, jj = np.indices((n, m))
        cost = np.where(np.abs(ii - jj) <= band, cost, np.inf)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    # Sweep anti-diagonals (i + j = d): every cell on a diagonal depends only on
    # the two previous diagonals, so each diagonal is one vectorised update.
    for d in range(2, n + m + 1):
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        j = d - i
        best = np.minimum(np.minimum(acc[i - 1, j], acc[i, j - 1]), acc[i - 1, j - 1])
        acc[i, j] = cost[i - 1, j - 1] + best
    dist = float(acc[n, m] / (n + m))
    if not return_path:
        return dist
    path = [(n - 1, m - 1)]
    i, j = n, m
    while (i, j) != (1, 1):
        steps = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        i, j = min(steps, key=lambda s: acc[s])
        path.append((i - 1, j - 1))
    return dist, path[::-1]
