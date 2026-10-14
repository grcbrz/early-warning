"""Aggregate fine rasters onto the 0.25 deg analysis grid."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from congo_ews.config import Grid

log = logging.getLogger(__name__)


def block_mean(a: np.ndarray, factor: int, min_valid_frac: float = 0.5) -> np.ndarray:
    """Mean over non-overlapping ``factor`` x ``factor`` blocks, NaN-aware."""
    if a.shape[0] % factor or a.shape[1] % factor:
        raise ValueError(f"shape {a.shape} not divisible by {factor}")
    blocks = a.reshape(a.shape[0] // factor, factor, a.shape[1] // factor, factor)
    valid = np.isfinite(blocks).sum(axis=(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks (ocean) -> NaN
        mean = np.nanmean(blocks, axis=(1, 3))
    mean[valid < min_valid_frac * factor * factor] = np.nan
    return mean.astype(np.float32)


def hansen_strip_counts(
    treecover: np.ndarray,
    lossyear: np.ndarray,
    block: int,
    canopy_threshold: int,
    n_years: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forest-pixel counts and per-year loss counts for each ``block``-wide cell of a strip.

    ``treecover``/``lossyear`` are (block, width) arrays. Loss is only counted on pixels that
    were forest in 2000, so the loss fraction is relative to the 2000 forest baseline.
    Returns ``forest`` (n_cells,) and ``loss`` (n_cells, n_years), year 1 at column 0.
    """
    n_cells = treecover.shape[1] // block
    cell_of_col = np.arange(treecover.shape[1]) // block
    cell = np.broadcast_to(cell_of_col, treecover.shape)
    forest_mask = treecover >= canopy_threshold
    forest = np.bincount(cell[forest_mask], minlength=n_cells)
    lost = forest_mask & (lossyear > 0) & (lossyear <= n_years)
    idx = cell[lost].astype(np.int64) * n_years + (lossyear[lost].astype(np.int64) - 1)
    loss = np.bincount(idx, minlength=n_cells * n_years).reshape(n_cells, n_years)
    return forest, loss


def aggregate_hansen_tile(
    treecover_path: Path,
    lossyear_path: Path,
    grid: Grid,
    canopy_threshold: int,
    n_years: int,
    forest: np.ndarray,
    loss: np.ndarray,
    total: np.ndarray,
) -> None:
    """Accumulate one tile's counts into grid-shaped ``forest``/``loss``/``total`` in place."""
    with rasterio.open(treecover_path) as tc_src, rasterio.open(lossyear_path) as ly_src:
        res = tc_src.res[0]
        block = int(round(grid.res / res))
        b, tb = grid.bbox, tc_src.bounds
        west, east = max(b.lon_min, tb.left), min(b.lon_max, tb.right)
        south, north = max(b.lat_min, tb.bottom), min(b.lat_max, tb.top)
        if west >= east or south >= north:
            return
        win = from_bounds(west, south, east, north, tc_src.transform).round_offsets().round_lengths()
        row0 = int(round((b.lat_max - north) / grid.res))
        col0 = int(round((west - b.lon_min) / grid.res))
        n_cell_cols = int(win.width) // block
        for i in range(int(win.height) // block):
            strip = Window(win.col_off, win.row_off + i * block, win.width, block)
            tc = tc_src.read(1, window=strip)
            ly = ly_src.read(1, window=strip)
            f, l = hansen_strip_counts(tc, ly, block, canopy_threshold, n_years)
            r = row0 + i
            forest[r, col0: col0 + n_cell_cols] += f
            loss[:, r, col0: col0 + n_cell_cols] += l.T
            total[r, col0: col0 + n_cell_cols] += block * block
