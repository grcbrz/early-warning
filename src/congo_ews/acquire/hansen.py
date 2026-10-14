"""Hansen et al. Global Forest Change (UMD) tiles: tree cover 2000 and loss year.

Tiles are 10x10 deg GeoTIFFs at 1 arc-second (~30 m), named by their north-west corner.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from congo_ews.config import BBox, CONGO_BBOX, PATHS, SOURCES

log = logging.getLogger(__name__)
LAYERS = ("treecover2000", "lossyear")


def tile_name(lat_top: int, lon_left: int) -> str:
    ns = f"{abs(lat_top):02d}{'N' if lat_top >= 0 else 'S'}"
    ew = f"{abs(lon_left):03d}{'E' if lon_left >= 0 else 'W'}"
    return f"{ns}_{ew}"


def tiles_for_bbox(bbox: BBox) -> list[str]:
    """Names of every 10 deg tile intersecting ``bbox``."""
    tops = range(int(math.ceil(bbox.lat_max / 10) * 10), int(math.floor(bbox.lat_min / 10) * 10), -10)
    lefts = range(int(math.floor(bbox.lon_min / 10) * 10), int(math.ceil(bbox.lon_max / 10) * 10), 10)
    return [tile_name(t, l) for t in tops for l in lefts]


def tile_url(layer: str, tile: str) -> str:
    return f"{SOURCES.hansen_base}/Hansen_{SOURCES.hansen_version}_{layer}_{tile}.tif"


def download(url: str, out: Path, chunk: int = 2**23) -> Path:
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk):
                fh.write(block)
    tmp.replace(out)
    return out


def fetch_all(bbox: BBox = CONGO_BBOX, workers: int = 4) -> list[Path]:
    out_dir = PATHS.raw / "hansen"
    jobs = [(tile_url(layer, t), out_dir / f"{layer}_{t}.tif")
            for t in tiles_for_bbox(bbox) for layer in LAYERS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: download(*j), jobs))
