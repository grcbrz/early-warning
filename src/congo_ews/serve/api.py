"""FastAPI service. Run: ``uvicorn congo_ews.serve.api:app``.

Environment: ``CONGO_EWS_ROOT`` (repo root with data/ and artifacts/), ``HANSEN_DIR``
(optional local Hansen tiles; otherwise tiles are range-read from Google Cloud Storage).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from congo_ews.config import ANALYSIS_GRID, CONGO_BBOX, PATHS
from congo_ews.serve.plot_check import check_plot
from congo_ews.serve.schemas import (
    CellDetail,
    CellRisk,
    IndicatorPoint,
    PlotCheckRequest,
    PlotCheckResponse,
)

DISCLAIMER = (
    "Screening only. Hansen GFC 'loss' is any stand-replacing tree-cover loss (including natural "
    "disturbance) at 30 m and is not the EUDR legal definition of deforestation. Risk scores are "
    "statistical forecasts for ~28 km cells, not plot-level determinations."
)


@dataclass
class Store:
    panel: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict
    hansen_dir: Path | None

    @classmethod
    def from_root(cls, root: Path, hansen_dir: Path | None) -> "Store":
        panel = pd.read_parquet(root / "data/processed/panel.parquet")
        preds = pd.read_parquet(root / "artifacts/predictions.parquet")
        preds["risk_percentile"] = preds.groupby("year")["risk"].rank(pct=True) * 100
        metrics = json.loads((root / "artifacts/metrics.json").read_text())
        return cls(panel, preds, metrics, hansen_dir)

    @property
    def latest_year(self) -> int:
        return int(self.predictions["year"].max())

    def cell_risks(self, cell_ids: list[int], year: int) -> list[CellRisk]:
        p = self.predictions[(self.predictions.year == year) & self.predictions.cell_idx.isin(cell_ids)]
        loc = self.panel.drop_duplicates("cell_idx").set_index("cell_idx")[["lat", "lon"]]
        return [CellRisk(cell_idx=int(r.cell_idx), lat=float(loc.at[r.cell_idx, "lat"]),
                         lon=float(loc.at[r.cell_idx, "lon"]), year=int(r.year),
                         risk=round(float(r.risk), 4), risk_percentile=round(float(r.risk_percentile), 1))
                for r in p.itertuples()]


def _nan_to_none(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def create_app(store: Store) -> FastAPI:
    app = FastAPI(title="Congo forest early-warning API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": store.metrics.get("selected"), "latest_year": store.latest_year,
                "n_cells": int(store.panel.cell_idx.nunique())}

    @app.get("/risk", response_model=list[CellRisk])
    def risk(year: int | None = None, top_k: int = Query(20, ge=1, le=5000)) -> list[CellRisk]:
        y = year if year is not None else store.latest_year
        p = store.predictions[store.predictions.year == y]
        if p.empty:
            raise HTTPException(404, f"no predictions for year {y}")
        ranked = store.cell_risks(p.nlargest(top_k, "risk").cell_idx.tolist(), y)
        return sorted(ranked, key=lambda c: -c.risk)

    @app.get("/cells/{cell_idx}", response_model=CellDetail)
    def cell(cell_idx: int) -> CellDetail:
        rows = store.panel[store.panel.cell_idx == cell_idx]
        if rows.empty:
            raise HTTPException(404, f"cell {cell_idx} is not an in-scope forest cell")
        rows = rows.merge(store.predictions[["year", "cell_idx", "risk"]], on=["year", "cell_idx"], how="left")
        history = [IndicatorPoint(year=int(r.year), ar1=_nan_to_none(r.ar1), variance=_nan_to_none(r.variance),
                                  ar1_tau=_nan_to_none(r.ar1_tau), loss_t=float(r.loss_t),
                                  label=_nan_to_none(r.label), risk=_nan_to_none(r.risk))
                   for r in rows.itertuples()]
        return CellDetail(cell_idx=cell_idx, lat=float(rows.lat.iloc[0]), lon=float(rows.lon.iloc[0]),
                          history=history)

    @app.post("/plot-check", response_model=PlotCheckResponse)
    def plot_check(req: PlotCheckRequest) -> PlotCheckResponse:
        ring = np.asarray(req.geometry.coordinates[0])
        b = CONGO_BBOX
        if not (b.lon_min <= ring[:, 0].min() and ring[:, 0].max() <= b.lon_max
                and b.lat_min <= ring[:, 1].min() and ring[:, 1].max() <= b.lat_max):
            raise HTTPException(422, "plot lies outside the supported region (lat -10..6, lon 8..32)")
        result = check_plot(req.geometry.model_dump(), store.hansen_dir)
        g = ANALYSIS_GRID
        rows = ((b.lat_max - ring[:, 1]) // g.res).astype(int).clip(0, g.n_rows - 1)
        cols = ((ring[:, 0] - b.lon_min) // g.res).astype(int).clip(0, g.n_cols - 1)
        cells = sorted({int(g.cell_id(r, c)) for r in range(rows.min(), rows.max() + 1)
                        for c in range(cols.min(), cols.max() + 1)})
        return PlotCheckResponse(
            plot_id=req.plot_id, **result.__dict__,
            flag="loss_after_cutoff" if result.loss_after_cutoff_ha > 0 else "no_loss_detected",
            containing_cells=store.cell_risks(cells, store.latest_year), disclaimer=DISCLAIMER)

    return app


def _default_app() -> FastAPI:
    root = Path(os.environ.get("CONGO_EWS_ROOT", PATHS.root))
    hansen = os.environ.get("HANSEN_DIR")
    return create_app(Store.from_root(root, Path(hansen) if hansen else None))


app = _default_app() if os.environ.get("CONGO_EWS_LAZY") != "1" else None
