"""Acceptance criteria from SPEC.md §6, checked against the artifacts built from real data.

Run after `make all`. Skipped when artifacts are absent (e.g. on a fresh clone in CI).
"""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from congo_ews.config import PATHS
from congo_ews.features import FEATURE_COLUMNS

pytestmark = pytest.mark.acceptance
ART, PROC = PATHS.artifacts, PATHS.processed
needs_artifacts = pytest.mark.skipif(not (ART / "metrics.json").exists(), reason="run `make all` first")


@needs_artifacts
def test_ac1_feature_table_reproducible():
    from congo_ews.pipeline import compute_features, panel_hash
    from congo_ews.features import SPLITS, add_labels

    load = lambda n: dict(np.load(PATHS.interim / f"{n}.npz"))
    panel, _ = add_labels(compute_features(load("hansen"), load("ndvi"), load("precip")), SPLITS["train"])
    meta = json.loads((PROC / "panel_meta.json").read_text())
    assert panel_hash(panel) == meta["panel_sha256"]


@needs_artifacts
def test_ac2_validation_report_clean():
    rep = json.loads((ART / "validation_report.json").read_text())
    for src in ("ndvi", "precip"):
        assert rep[src]["n_months"] == 300 and rep[src]["missing_months"] == []
    assert -1.0 <= rep["ndvi"]["value_min"] and rep["ndvi"]["value_max"] <= 1.0


@needs_artifacts
def test_ac5_models_scored_and_selection_rule():
    m = json.loads((ART / "metrics.json").read_text())
    names = {"B0_persistence", "B1_logreg_pressure", "M1_hgb_pressure", "M2_hgb_all"}
    assert set(m["val"]) == names and set(m["test"]) == names
    assert m["val"][m["selected"]]["pr_auc"] >= m["val"]["B0_persistence"]["pr_auc"]
    assert m["split_years"]["val"][1] < m["split_years"]["test"][0]


@needs_artifacts
def test_ac7_drift_report_has_thresholds_and_all_features():
    d = json.loads((ART / "drift_report.json").read_text())
    for ref in ("vs_training", "vs_recent"):
        assert d[ref]["thresholds"] == {"warn": 0.1, "alert": 0.25}
        assert set(d[ref]["features"]) == set(FEATURE_COLUMNS)


@needs_artifacts
def test_ac6_api_latency_on_real_store():
    os.environ["CONGO_EWS_LAZY"] = "1"
    from fastapi.testclient import TestClient
    from congo_ews.serve.api import Store, create_app

    client = TestClient(create_app(Store.from_root(PATHS.root, None)))
    cells = pd.read_parquet(ART / "predictions.parquet").cell_idx.unique()[:50]
    times = []
    for c in cells:
        t = time.perf_counter()
        assert client.get(f"/cells/{int(c)}").status_code == 200
        times.append(time.perf_counter() - t)
    assert np.percentile(times, 95) < 0.3
