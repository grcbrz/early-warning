"""Integration: full feature + model pipeline on a small synthetic region (AC1, AC4)."""

import numpy as np
import pandas as pd
import pytest

from congo_ews.config import BBox, Grid
from congo_ews.features import FEATURE_COLUMNS, SPLITS, add_labels, split_by_year
from congo_ews.model.train import run_experiment
from congo_ews.pipeline import compute_features, panel_hash

GRID = Grid(bbox=BBox(0.0, 2.0, 0.0, 3.0), res=0.25)  # 8 x 12 cells


def synthetic_inputs(seed: int = 0) -> tuple[dict, dict, dict]:
    rng = np.random.default_rng(seed)
    r, c = GRID.n_rows, GRID.n_cols
    years = np.arange(2001, 2025)
    forest = np.clip(rng.uniform(0.3, 1.0, (r, c)), 0, 1).astype(np.float32)
    # Loss pressure spreads from the west edge over time, plus noise.
    pressure = np.exp(-np.arange(c) / 3.0)[None, :] * np.ones((r, 1))
    loss = np.stack([0.002 * pressure * (1 + 0.1 * (y - 2001)) + rng.exponential(0.0005, (r, c))
                     for y in years]).astype(np.float32)
    months = pd.date_range("2001-01-01", "2025-12-01", freq="MS")
    seasonal = 0.05 * np.sin(2 * np.pi * months.month.to_numpy() / 12)[:, None, None]
    ndvi = (0.8 + seasonal + 0.02 * rng.standard_normal((len(months), r, c))).astype(np.float32)
    ndvi[rng.random(ndvi.shape) < 0.1] = np.nan  # cloud gaps
    sensor = np.where(months.year < 2014, "AVHRR", "VIIRS")
    precip = rng.gamma(4.0, 40.0, (len(months), r, c)).astype(np.float32)
    hansen = {"forest_frac_2000": forest, "loss_frac": loss, "loss_years": years}
    return (hansen,
            {"ndvi": ndvi, "months": months.strftime("%Y-%m").to_numpy(), "sensor": sensor},
            {"precip": precip, "months": months.strftime("%Y-%m").to_numpy()})


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return compute_features(*synthetic_inputs(), grid=GRID)


def test_panel_schema_and_scope(panel):
    assert set(FEATURE_COLUMNS) <= set(panel.columns)
    assert panel[["year", "cell_idx"]].duplicated().sum() == 0
    assert panel.year.min() == 2003 and panel.year.max() == 2024
    assert panel.loc[panel.year == 2024, "target_loss"].isna().all()
    # Includes windows around the 2014 sensor switch, where a short era segment is dropped.
    assert panel.loc[panel.year >= 2005, "ar1"].notna().mean() > 0.95


def test_deterministic_feature_hash(panel):
    """AC1: identical inputs give a byte-identical feature table."""
    again = compute_features(*synthetic_inputs(), grid=GRID)
    assert panel_hash(panel) == panel_hash(again)


@pytest.mark.parametrize("cutoff", [2010, 2016])
def test_no_temporal_leakage(panel, cutoff):
    """AC4: features at year t are unchanged when every observation after Dec t is removed."""
    hansen, ndvi, precip = synthetic_inputs()
    keep_y = hansen["loss_years"] <= cutoff
    hansen_c = {**hansen, "loss_frac": hansen["loss_frac"][keep_y], "loss_years": hansen["loss_years"][keep_y]}
    keep_m = ndvi["months"] <= f"{cutoff}-12"
    ndvi_c = {k: (v[keep_m] if k != "sensor" else v[keep_m]) for k, v in ndvi.items()}
    precip_c = {k: v[keep_m] for k, v in precip.items()}
    truncated = compute_features(hansen_c, ndvi_c, precip_c, grid=GRID)
    a = panel[panel.year == cutoff].set_index("cell_idx")[FEATURE_COLUMNS]
    b = truncated[truncated.year == cutoff].set_index("cell_idx")[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(a, b)


def test_experiment_runs_end_to_end(panel):
    labelled, thr = add_labels(panel, SPLITS["train"])
    splits = split_by_year(labelled)
    for k in ("train", "val", "test"):
        splits[k] = splits[k][splits[k]["label"].notna()]
    results, preds, best, model = run_experiment(splits, n_boot=50)
    assert results["selected"] in results["val"]
    assert set(preds.split) == {"test", "live"}
    assert preds.risk.between(0, 1).all()
    # Pressure gradient is learnable: the selected model must beat a coin flip on test.
    assert results["test"][best.name]["roc_auc"] > 0.6
