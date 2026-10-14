"""PERSIANN-CDR daily precipitation (0.25 deg, 1983-) from AWS Open Data -> monthly totals."""

from __future__ import annotations

import calendar
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from congo_ews.acquire.s3 import get_with_retry, list_keys
from congo_ews.config import BBox, CONGO_BBOX, PATHS, SOURCES

log = logging.getLogger(__name__)
_DATE_RE = re.compile(r"_(\d{8})_c\d+\.nc$")


def read_daily(content: bytes, bbox: BBox) -> np.ndarray:
    """Return daily precipitation (mm) on a north-up (lat desc, lon asc) grid."""
    with h5py.File(io.BytesIO(content), "r") as h5:
        lat, lon = h5["lat"][:], h5["lon"][:]
        ds = h5["precipitation"]
        data = ds[0].astype(np.float32)  # stored (lon, lat)
        fill = ds.attrs.get("_FillValue", [-9999.0])[0]
    data[(data == fill) | (data < 0)] = np.nan
    lon = np.where(lon > 180, lon - 360, lon)
    rows = np.where((lat <= bbox.lat_max) & (lat >= bbox.lat_min))[0]
    cols = np.where((lon >= bbox.lon_min) & (lon <= bbox.lon_max))[0]
    sub = data[np.ix_(cols, rows)].T  # -> (lat, lon)
    lat_sub, lon_sub = lat[rows], lon[cols]
    sub = sub[np.argsort(-lat_sub)][:, np.argsort(lon_sub)]
    return sub


def fetch_month(year: int, month: int, out_dir: Path, bbox: BBox = CONGO_BBOX) -> Path | None:
    out = out_dir / f"{year:04d}-{month:02d}.npz"
    if out.exists():
        return out
    prefix = f"data/{year:04d}/"
    want = f"{year:04d}{month:02d}"
    keys = [k for k in list_keys(SOURCES.precip_bucket, prefix)
            if (m := _DATE_RE.search(k)) and m.group(1).startswith(want)]
    n_days = calendar.monthrange(year, month)[1]
    if len(keys) < n_days:
        log.warning("%d-%02d: only %d/%d daily files", year, month, len(keys), n_days)
    if not keys:
        return None
    days = [read_daily(get_with_retry(f"{SOURCES.precip_bucket}/{k}").content, bbox)
            for k in keys]
    stack = np.stack(days)
    # Scale the observed-day mean to the full month so a missing day does not bias totals.
    total = np.nanmean(stack, axis=0) * n_days
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, precip=total.astype(np.float32), n_days=len(keys))
    tmp.replace(out)
    return out


def fetch_range(start_year: int, end_year: int, workers: int = 16) -> list[Path]:
    out_dir = PATHS.raw / "precip"
    months = [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda ym: fetch_month(*ym, out_dir), months))
    return [r for r in results if r is not None]
