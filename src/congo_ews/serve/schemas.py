"""Request/response contracts for the API (SPEC §5, AC6)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_PLOT_SPAN_DEG = 0.2  # ~22 km: bounds the 30 m read to a few MB per request


class PolygonGeometry(BaseModel):
    type: Literal["Polygon"]
    coordinates: list[list[tuple[float, float]]] = Field(min_length=1)

    @field_validator("coordinates")
    @classmethod
    def _valid_ring(cls, rings: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
        ring = rings[0]
        if len(ring) < 4:
            raise ValueError("outer ring needs at least 4 positions")
        if ring[0] != ring[-1]:
            raise ValueError("outer ring must be closed (first position == last)")
        for lon, lat in ring:
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f"position out of range: {lon}, {lat}")
        lons, lats = [p[0] for p in ring], [p[1] for p in ring]
        if max(lons) - min(lons) > MAX_PLOT_SPAN_DEG or max(lats) - min(lats) > MAX_PLOT_SPAN_DEG:
            raise ValueError(f"plot larger than {MAX_PLOT_SPAN_DEG} deg; split it")
        if max(lons) == min(lons) or max(lats) == min(lats):
            raise ValueError("degenerate polygon")
        return rings


class PlotCheckRequest(BaseModel):
    geometry: PolygonGeometry
    plot_id: str | None = None


class CellRisk(BaseModel):
    cell_idx: int
    lat: float
    lon: float
    year: int = Field(description="Feature year t; risk refers to year t+1")
    risk: float
    risk_percentile: float


class PlotCheckResponse(BaseModel):
    plot_id: str | None
    area_ha: float
    forest_2000_ha: float
    loss_2001_2020_ha: float
    loss_after_cutoff_ha: float
    loss_after_cutoff_by_year_ha: dict[int, float]
    flag: Literal["no_loss_detected", "loss_after_cutoff"]
    containing_cells: list[CellRisk]
    disclaimer: str


class IndicatorPoint(BaseModel):
    year: int
    ar1: float | None
    variance: float | None
    ar1_tau: float | None
    loss_t: float
    label: float | None
    risk: float | None


class CellDetail(BaseModel):
    cell_idx: int
    lat: float
    lon: float
    history: list[IndicatorPoint]
