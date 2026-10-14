"""Static configuration: study region, analysis grid, data sources and paths.

Everything that a reviewer might want to change lives here, so the rest of the
code base never hard-codes a coordinate, URL or year.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box in degrees (WGS84)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-90 <= self.lat_min < self.lat_max <= 90):
            raise ValueError(f"invalid latitude range: {self.lat_min}..{self.lat_max}")
        if not (-180 <= self.lon_min < self.lon_max <= 180):
            raise ValueError(f"invalid longitude range: {self.lon_min}..{self.lon_max}")


@dataclass(frozen=True)
class Grid:
    """Regular lat/lon analysis grid. Cell (row 0, col 0) is the north-west corner."""

    bbox: BBox
    res: float

    @property
    def n_rows(self) -> int:
        return int(round((self.bbox.lat_max - self.bbox.lat_min) / self.res))

    @property
    def n_cols(self) -> int:
        return int(round((self.bbox.lon_max - self.bbox.lon_min) / self.res))

    @property
    def lat_centers(self) -> np.ndarray:
        return self.bbox.lat_max - self.res * (np.arange(self.n_rows) + 0.5)

    @property
    def lon_centers(self) -> np.ndarray:
        return self.bbox.lon_min + self.res * (np.arange(self.n_cols) + 0.5)

    def cell_id(self, row: np.ndarray | int, col: np.ndarray | int) -> np.ndarray | int:
        return row * self.n_cols + col


# Congo Basin humid forest: Cameroon, Gabon, Eq. Guinea, Rep. Congo, CAR (south), DRC.
CONGO_BBOX = BBox(lat_min=-10.0, lat_max=6.0, lon_min=8.0, lon_max=32.0)
ANALYSIS_GRID = Grid(bbox=CONGO_BBOX, res=0.25)


@dataclass(frozen=True)
class Sources:
    ndvi_bucket: str = "https://noaa-cdr-ndvi-pds.s3.amazonaws.com"
    precip_bucket: str = "https://noaa-cdr-precip-persiann-pds.s3.amazonaws.com"
    hansen_base: str = (
        "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12"
    )
    hansen_version: str = "GFC-2024-v1.12"
    # Days of month sampled for the NDVI maximum-value composite.
    ndvi_sample_days: tuple[int, ...] = (1, 5, 9, 13, 17, 21, 25, 29)


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"


START_YEAR = 2001
END_YEAR = 2025
# Hansen GFC v1.12 reports loss for 2001..2024.
LOSS_LAST_YEAR = 2024
# A cell is "forest" if at least this share of its area had >= 30 % canopy in 2000.
FOREST_CANOPY_THRESHOLD = 30
MIN_FOREST_FRACTION = 0.5

SOURCES = Sources()
PATHS = Paths()
