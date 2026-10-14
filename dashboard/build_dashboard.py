"""Render the static results dashboard from pipeline artifacts: python dashboard/build_dashboard.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from congo_ews.config import ANALYSIS_GRID, PATHS

HERE = Path(__file__).parent
AR1_MAP_YEAR = 2022  # last window (2018-2022) inside a single sensor era (VIIRS S-NPP)


def grid_layer(df: pd.DataFrame, col: str, digits: int = 4) -> list[float | None]:
    g = ANALYSIS_GRID
    out: list[float | None] = [None] * (g.n_rows * g.n_cols)
    for cid, v in zip(df.cell_idx.to_numpy(), df[col].to_numpy()):
        if np.isfinite(v):
            out[int(cid)] = round(float(v), digits)
    return out


def collect() -> dict:
    panel = pd.read_parquet(PATHS.processed / "panel.parquet")
    preds = pd.read_parquet(PATHS.artifacts / "predictions.parquet")
    metrics = json.loads((PATHS.artifacts / "metrics.json").read_text())
    drift = json.loads((PATHS.artifacts / "drift_report.json").read_text())
    hansen = np.load(PATHS.interim / "hansen.npz")
    forest = hansen["forest_frac_2000"] >= 0.5
    loss = hansen["loss_frac"][:, forest]
    live = preds[preds.split == "live"]
    live_year = int(live.year.iloc[0])
    loss_last = panel[panel.year == live_year].assign(pct=lambda d: d.loss_t * 100)
    ar1 = panel[panel.year == AR1_MAP_YEAR]
    yearly = panel.groupby("year").agg(ar1=("ar1", "median"), variance=("variance", "median")).reset_index()
    top = live.nlargest(10, "risk").merge(
        panel[panel.year == live_year][["cell_idx", "lat", "lon", "loss_t", "cum_loss"]], on="cell_idx")
    return {
        "grid": {"rows": ANALYSIS_GRID.n_rows, "cols": ANALYSIS_GRID.n_cols, "res": ANALYSIS_GRID.res,
                 "lat_max": ANALYSIS_GRID.bbox.lat_max, "lon_min": ANALYSIS_GRID.bbox.lon_min},
        "layers": {
            "risk": grid_layer(live, "risk"),
            "loss": grid_layer(loss_last, "pct"),
            "ar1": grid_layer(ar1, "ar1"),
        },
        "live_year": live_year,
        "ar1_year": AR1_MAP_YEAR,
        "loss_series": [{"year": int(y), "pct": round(float(loss[i].mean() * 100), 3)}
                        for i, y in enumerate(hansen["loss_years"])],
        "ar1_series": [{"year": int(r.year), "ar1": None if np.isnan(r.ar1) else round(float(r.ar1), 4)}
                       for r in yearly.itertuples()],
        "metrics": metrics,
        "drift": drift,
        "top": [{"cell": int(r.cell_idx), "lat": float(r.lat), "lon": float(r.lon), "risk": round(float(r.risk), 3),
                 "loss": round(float(r.loss_t) * 100, 2), "cum": round(float(r.cum_loss) * 100, 1)}
                for r in top.itertuples()],
        "n_cells": int(panel.cell_idx.nunique()),
    }


def main() -> None:
    template = (HERE / "template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(collect(), separators=(",", ":")))
    (HERE / "index.html").write_text(html)
    print(f"wrote {HERE / 'index.html'} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
