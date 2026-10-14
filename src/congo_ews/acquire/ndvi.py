"""NOAA Climate Data Record NDVI (AVHRR 1981-2013, VIIRS 2014-) from AWS Open Data.

Each daily file is a global 0.05 deg netCDF4 (~60 MB). We never download a whole
file: h5py reads only the gzip chunks intersecting the study box via HTTP range
requests (~1.5 s per file). Days are composited per month with the standard
maximum-value composite (Holben 1986), which suppresses cloud contamination.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import dataclass
from pathlib import Path

import fsspec
import h5py
import numpy as np

from congo_ews.acquire.s3 import list_keys
from congo_ews.config import BBox, CONGO_BBOX, PATHS, SOURCES

log = logging.getLogger(__name__)

_DATE_RE = re.compile(r"_(\d{8})_c\d+\.nc$")
_SENSOR_RE = re.compile(r"^data/\d{4}/(AVHRR|VIIRS)-Land_v\d+_[A-Z0-9]+_([A-Za-z0-9-]+)_\d{8}_")


@dataclass(frozen=True)
class NdviKey:
    key: str
    date: str  # YYYYMMDD
    sensor: str  # AVHRR | VIIRS
    platform: str  # e.g. NOAA-19, S-NPP, NOAA-20


def parse_key(key: str) -> NdviKey | None:
    date, sensor = _DATE_RE.search(key), _SENSOR_RE.match(key)
    if not (date and sensor):
        return None
    return NdviKey(key=key, date=date.group(1), sensor=sensor.group(1), platform=sensor.group(2))


def keys_for_month(year: int, month: int, days: tuple[int, ...]) -> list[NdviKey]:
    wanted = {f"{year:04d}{month:02d}{d:02d}" for d in days}
    parsed = (parse_key(k) for k in list_keys(SOURCES.ndvi_bucket, f"data/{year:04d}/"))
    # If a date has several reprocessed versions, keep the last (newest) listing.
    by_date = {p.date: p for p in parsed if p is not None and p.date in wanted}
    return sorted(by_date.values(), key=lambda p: p.date)


def read_subset(url: str, bbox: BBox) -> np.ndarray:
    """Read scaled NDVI for ``bbox``; invalid / fill values become NaN."""
    with fsspec.open(url, block_size=2**22).open() as fh, h5py.File(fh, "r") as h5:
        lat, lon = h5["latitude"][:], h5["longitude"][:]
        rows = np.where((lat <= bbox.lat_max) & (lat >= bbox.lat_min))[0]
        cols = np.where((lon >= bbox.lon_min) & (lon <= bbox.lon_max))[0]
        ds = h5["NDVI"]
        raw = ds[0, rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].astype(np.float32)
        fill = ds.attrs["_FillValue"][0]
        vmin, vmax = ds.attrs["valid_range"]
        scale = float(ds.attrs["scale_factor"][0])
        offset = float(ds.attrs.get("add_offset", [0.0])[0])
    raw[(raw == fill) | (raw < vmin) | (raw > vmax)] = np.nan
    return raw * scale + offset


def max_value_composite(stack: np.ndarray) -> np.ndarray:
    """Per-pixel maximum over the first axis, NaN where every observation is missing."""
    if stack.ndim != 3:
        raise ValueError(f"expected (time, y, x) stack, got shape {stack.shape}")
    out = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    valid = ~np.all(np.isnan(stack), axis=0)
    out[valid] = np.nanmax(stack[:, valid], axis=0)
    return out


def fetch_month(year: int, month: int, out_dir: Path, bbox: BBox = CONGO_BBOX) -> Path | None:
    """Download, composite and cache one month. Idempotent: skips existing output."""
    out = out_dir / f"{year:04d}-{month:02d}.npz"
    if out.exists():
        return out
    keys = keys_for_month(year, month, SOURCES.ndvi_sample_days)
    if not keys:
        log.warning("no NDVI files for %d-%02d", year, month)
        return None
    layers = []
    for k in keys:
        try:
            layers.append(read_subset(f"{SOURCES.ndvi_bucket}/{k.key}", bbox))
        except (OSError, KeyError) as exc:  # corrupt / partial file: skip the day
            log.warning("skipping %s: %s", k.key, exc)
    if not layers:
        return None
    composite = max_value_composite(np.stack(layers))
    sensors = sorted({k.sensor for k in keys})
    platforms = sorted({k.platform for k in keys})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ndvi=composite, n_obs=len(layers), sensor=",".join(sensors),
                        platform=",".join(platforms))
    tmp.replace(out)
    return out


def _fetch(ym: tuple[int, int], out_dir: Path) -> Path | None:
    return fetch_month(ym[0], ym[1], out_dir)


def fetch_range(start_year: int, end_year: int, workers: int = 16) -> list[Path]:
    # Processes, not threads: h5py holds a global lock, so threads would serialise reads.
    out_dir = PATHS.raw / "ndvi"
    months = [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(partial(_fetch, out_dir=out_dir), months))
    return [r for r in results if r is not None]
