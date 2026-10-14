"""EUDR-style screening of one plot against Hansen 30 m loss (cut-off date 31 Dec 2020).

Screening only: Hansen "loss" is any stand-replacement disturbance (incl. natural), not the
EUDR legal definition of deforestation, and has no forest-type or land-use attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

from congo_ews.acquire.hansen import tile_name, tile_url
from congo_ews.config import FOREST_CANOPY_THRESHOLD

EUDR_CUTOFF_LOSSYEAR = 20  # lossyear codes 1..24 = 2001..2024; >20 means after 31 Dec 2020
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


@dataclass(frozen=True)
class PlotResult:
    area_ha: float
    forest_2000_ha: float
    loss_2001_2020_ha: float
    loss_after_cutoff_ha: float
    loss_after_cutoff_by_year_ha: dict[int, float]


def tiles_for_bounds(west: float, south: float, east: float, north: float) -> list[str]:
    tops = range(int(math.ceil(north / 10) * 10), int(math.ceil(south / 10) * 10) - 10, -10)
    lefts = range(int(math.floor(west / 10) * 10), int(math.floor(east / 10) * 10) + 10, 10)
    return [tile_name(t, l) for t in tops for l in lefts]


def _source(layer: str, tile: str, hansen_dir: Path | None) -> str:
    if hansen_dir is not None and (hansen_dir / f"{layer}_{tile}.tif").exists():
        return str(hansen_dir / f"{layer}_{tile}.tif")
    return "/vsicurl/" + tile_url(layer, tile)


def pixel_area_ha(lat: np.ndarray, res_deg: float) -> np.ndarray:
    """Area of a res x res degree pixel centred at ``lat`` (spherical approximation)."""
    return (res_deg * M_PER_DEG_LAT) * (res_deg * M_PER_DEG_LON_EQ * np.cos(np.radians(lat))) / 1e4


def _outer_window(win: Window) -> Window:
    """Smallest integer pixel window containing ``win`` (so edge pixels are not dropped)."""
    c0, r0 = math.floor(win.col_off + 1e-9), math.floor(win.row_off + 1e-9)
    c1 = math.ceil(win.col_off + win.width - 1e-9)
    r1 = math.ceil(win.row_off + win.height - 1e-9)
    return Window(c0, r0, c1 - c0, r1 - r0)


def check_plot(geometry: dict, hansen_dir: Path | None = None) -> PlotResult:
    coords = np.asarray(geometry["coordinates"][0], dtype=float)
    west, south = coords.min(axis=0)
    east, north = coords.max(axis=0)
    totals = {"area": 0.0, "forest": 0.0, "pre": 0.0, "post": 0.0}
    by_year: dict[int, float] = {}
    for tile in tiles_for_bounds(west, south, east, north):
        with rasterio.open(_source("treecover2000", tile, hansen_dir)) as tc_src, \
             rasterio.open(_source("lossyear", tile, hansen_dir)) as ly_src:
            tb = tc_src.bounds
            w, s, e, n = max(west, tb.left), max(south, tb.bottom), min(east, tb.right), min(north, tb.top)
            if w >= e or s >= n:
                continue
            win = _outer_window(from_bounds(w, s, e, n, tc_src.transform))
            tc = tc_src.read(1, window=win, boundless=True, fill_value=0)
            ly = ly_src.read(1, window=win, boundless=True, fill_value=0)
            transform = tc_src.window_transform(win)
            res = tc_src.res[0]
        inside = geometry_mask([geometry], out_shape=tc.shape, transform=transform, invert=True)
        lat = transform.f + transform.e * (np.arange(tc.shape[0]) + 0.5)
        area = np.broadcast_to(pixel_area_ha(lat, res)[:, None], tc.shape)
        forest = inside & (tc >= FOREST_CANOPY_THRESHOLD)
        totals["area"] += area[inside].sum()
        totals["forest"] += area[forest].sum()
        totals["pre"] += area[forest & (ly >= 1) & (ly <= EUDR_CUTOFF_LOSSYEAR)].sum()
        post = forest & (ly > EUDR_CUTOFF_LOSSYEAR)
        totals["post"] += area[post].sum()
        for code in np.unique(ly[post]):
            yr = 2000 + int(code)
            by_year[yr] = by_year.get(yr, 0.0) + float(area[post & (ly == code)].sum())
    return PlotResult(
        area_ha=round(totals["area"], 3),
        forest_2000_ha=round(totals["forest"], 3),
        loss_2001_2020_ha=round(totals["pre"], 3),
        loss_after_cutoff_ha=round(totals["post"], 3),
        loss_after_cutoff_by_year_ha={k: round(v, 3) for k, v in sorted(by_year.items())},
    )
