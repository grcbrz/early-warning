"""End-to-end orchestration: `python -m congo_ews.pipeline {interim|features|train|all}`."""

from __future__ import annotations

import hashlib
import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from congo_ews.config import ANALYSIS_GRID, PATHS, Grid, Paths
from congo_ews.features import FEATURE_COLUMNS, add_labels, annual_climate, build_panel, split_by_year, SPLITS
from congo_ews.model.train import run_experiment
from congo_ews.monitoring.drift import drift_report
from congo_ews.process import build
from congo_ews.process.indicators import rolling_indicators

log = logging.getLogger("congo_ews")


def compute_features(hansen: dict, ndvi: dict, precip: dict, grid: Grid = ANALYSIS_GRID) -> pd.DataFrame:
    n_cells = grid.n_rows * grid.n_cols
    ndvi_months = pd.DatetimeIndex(pd.to_datetime(ndvi["months"]))
    indicators = rolling_indicators(ndvi["ndvi"].reshape(len(ndvi_months), n_cells), ndvi_months,
                                    np.asarray(ndvi["sensor"]))
    p_months = pd.DatetimeIndex(pd.to_datetime(precip["months"]))
    climate = annual_climate(precip["precip"].reshape(len(p_months), n_cells).astype(np.float64), p_months)
    return build_panel(hansen["forest_frac_2000"], hansen["loss_frac"], hansen["loss_years"],
                       indicators, climate, grid)


def panel_hash(panel: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(panel, index=True).values.tobytes()).hexdigest()


def stage_interim(paths: Paths) -> None:
    build.run(paths)


def stage_features(paths: Paths) -> pd.DataFrame:
    load = lambda name: dict(np.load(paths.interim / f"{name}.npz", allow_pickle=False))
    panel = compute_features(load("hansen"), load("ndvi"), load("precip"))
    panel, threshold = add_labels(panel, SPLITS["train"])
    paths.processed.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(paths.processed / "panel.parquet", index=False)
    meta = {"label_threshold": threshold, "panel_sha256": panel_hash(panel), "n_rows": len(panel),
            "n_cells": int(panel.cell_idx.nunique()), "years": [int(panel.year.min()), int(panel.year.max())]}
    (paths.processed / "panel_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("panel: %s", meta)
    return panel


def stage_train(paths: Paths, n_boot: int = 1000) -> dict:
    panel = pd.read_parquet(paths.processed / "panel.parquet")
    meta = json.loads((paths.processed / "panel_meta.json").read_text())
    splits = split_by_year(panel)
    for k in ("train", "val", "test"):
        splits[k] = splits[k][splits[k]["label"].notna()]
    results, preds, best, model = run_experiment(splits, n_boot=n_boot)
    results["label_threshold"] = meta["label_threshold"]
    results["panel_sha256"] = meta["panel_sha256"]
    results["split_years"] = {k: [min(v), max(v)] for k, v in SPLITS.items()}
    results["n_rows"] = {k: int(len(v)) for k, v in splits.items()}
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    (paths.artifacts / "metrics.json").write_text(json.dumps(results, indent=2))
    preds.to_parquet(paths.artifacts / "predictions.parquet", index=False)
    joblib.dump({"model": model, "features": best.features, "name": best.name}, paths.artifacts / "model.joblib")
    live_year = int(splits["live"].year.iloc[0])
    recent = panel[panel.year.between(live_year - 3, live_year - 1)]
    drift = {
        "current_year": live_year,
        # Trending features (cumulative loss) always drift against old training years, so the
        # actionable comparison is against the most recent labelled years.
        "vs_training": {"reference_years": [min(SPLITS["train"]), max(SPLITS["train"])],
                        **drift_report(splits["train"], splits["live"], FEATURE_COLUMNS)},
        "vs_recent": {"reference_years": [live_year - 3, live_year - 1],
                      **drift_report(recent, splits["live"], FEATURE_COLUMNS)},
    }
    (paths.artifacts / "drift_report.json").write_text(json.dumps(drift, indent=2))
    return results


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stage = argv[1] if len(argv) > 1 else "all"
    if stage in ("interim", "all"):
        stage_interim(PATHS)
    if stage in ("features", "all"):
        stage_features(PATHS)
    if stage in ("train", "all"):
        print(json.dumps(stage_train(PATHS), indent=2))


if __name__ == "__main__":
    main(sys.argv)
