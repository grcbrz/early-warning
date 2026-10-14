"""Turn raw downloads into analysis-grid arrays in ``data/interim``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from congo_ews.acquire.hansen import tiles_for_bbox
from congo_ews.config import (
    ANALYSIS_GRID,
    FOREST_CANOPY_THRESHOLD,
    LOSS_LAST_YEAR,
    PATHS,
    Grid,
)
from congo_ews.process.aggregate import aggregate_hansen_tile, block_mean
from congo_ews.process.validate import validate_monthly_stack

log = logging.getLogger(__name__)
NDVI_NATIVE_RES = 0.05


def build_hansen(raw_dir: Path, grid: Grid = ANALYSIS_GRID) -> dict[str, np.ndarray]:
    n_years = LOSS_LAST_YEAR - 2000
    forest = np.zeros((grid.n_rows, grid.n_cols), np.int64)
    total = np.zeros_like(forest)
    loss = np.zeros((n_years, grid.n_rows, grid.n_cols), np.int64)
    for tile in tiles_for_bbox(grid.bbox):
        log.info("aggregating Hansen tile %s", tile)
        aggregate_hansen_tile(raw_dir / f"treecover2000_{tile}.tif", raw_dir / f"lossyear_{tile}.tif",
                              grid, FOREST_CANOPY_THRESHOLD, n_years, forest, loss, total)
    if (total == 0).any():
        raise ValueError("some grid cells were not covered by any Hansen tile")
    with np.errstate(all="ignore"):
        loss_frac = np.where(forest > 0, loss / forest, 0.0)
    return {
        "forest_frac_2000": (forest / total).astype(np.float32),
        "loss_frac": loss_frac.astype(np.float32),  # (year, row, col), share of 2000 forest
        "loss_years": np.arange(2001, LOSS_LAST_YEAR + 1),
    }


def _load_monthly(raw_dir: Path, var: str) -> tuple[pd.DatetimeIndex, list[np.ndarray], list[dict]]:
    files = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].npz"))
    months, arrays, meta = [], [], []
    for f in files:
        with np.load(f) as z:
            months.append(pd.Timestamp(f.stem + "-01"))
            arrays.append(z[var])
            meta.append({k: z[k].item() for k in z.files if k != var})
    return pd.DatetimeIndex(months), arrays, meta


def sensor_era(sensor: str, platform: str) -> str:
    """Homogeneous-calibration era. AVHRR CDR v5 is inter-calibrated across NOAA platforms;
    VIIRS S-NPP -> NOAA-20 (Jan 2023) is not, so each VIIRS platform is its own era."""
    return "AVHRR" if sensor == "AVHRR" else f"{sensor}-{platform}"


def build_ndvi(raw_dir: Path, grid: Grid = ANALYSIS_GRID) -> tuple[dict[str, np.ndarray], dict]:
    months, arrays, meta = _load_monthly(raw_dir, "ndvi")
    factor = int(round(grid.res / NDVI_NATIVE_RES))
    native = np.stack(arrays)
    report = validate_monthly_stack(native, months, (grid.n_rows * factor, grid.n_cols * factor),
                                    value_range=(-1.0, 1.0), max_missing_months=6)
    coarse = np.stack([block_mean(a, factor, min_valid_frac=0.5) for a in arrays])
    full = pd.date_range(months.min(), months.max(), freq="MS")
    out = np.full((len(full), grid.n_rows, grid.n_cols), np.nan, np.float32)
    out[full.get_indexer(months)] = coarse
    sensor = pd.Series([sensor_era(m["sensor"], m.get("platform", "")) for m in meta],
                       index=months).reindex(full)
    # A month with no file inherits the era of its neighbours (needed for era-wise stats).
    sensor = sensor.ffill().bfill()
    report["sensors_by_year"] = sensor.groupby(full.year).agg(lambda s: sorted(set(s))).to_dict()
    return {"ndvi": out, "months": full.strftime("%Y-%m").to_numpy().astype(str),
            "sensor": sensor.to_numpy().astype(str)}, report


def build_precip(raw_dir: Path, grid: Grid = ANALYSIS_GRID) -> tuple[dict[str, np.ndarray], dict]:
    months, arrays, _ = _load_monthly(raw_dir, "precip")
    stack = np.stack(arrays)
    report = validate_monthly_stack(stack, months, (grid.n_rows, grid.n_cols),
                                    value_range=(0.0, 3000.0), max_missing_months=0)
    return {"precip": stack, "months": months.strftime("%Y-%m").to_numpy().astype(str)}, report


def run(paths=PATHS, force: bool = False) -> None:
    """Build every interim array; existing outputs are reused unless ``force``."""
    paths.interim.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    report_path = paths.artifacts / "validation_report.json"
    reports = json.loads(report_path.read_text()) if report_path.exists() else {}
    if force or not (paths.interim / "hansen.npz").exists():
        np.savez_compressed(paths.interim / "hansen.npz", **build_hansen(paths.raw / "hansen"))
    for name, builder in (("ndvi", build_ndvi), ("precip", build_precip)):
        if force or not (paths.interim / f"{name}.npz").exists():
            arrays, reports[name] = builder(paths.raw / name)
            np.savez_compressed(paths.interim / f"{name}.npz", **arrays)
    report_path.write_text(json.dumps(reports, indent=2, default=str))
