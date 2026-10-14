"""Metric and drift helpers."""

import numpy as np
import pandas as pd
import pytest

from congo_ews.model.evaluate import at_fraction, bootstrap_pr_auc_diff, score_metrics
from congo_ews.monitoring.drift import PSI_ALERT, drift_report, psi


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "year": [2020] * 10,
        "cell_idx": range(10),
        "label": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "perfect": np.linspace(1, 0, 10),
        "worst": np.linspace(0, 1, 10),
    })


def test_at_fraction_perfect_ranking_hits_ceiling():
    r = at_fraction(_df(), "perfect", frac=0.1)  # k = 1, 2 positives
    assert r["precision"] == 1.0 and r["recall"] == 0.5 and r["recall_ceiling"] == 0.5


def test_score_metrics_perfect_and_worst():
    assert score_metrics(_df(), "perfect")["pr_auc"] == pytest.approx(1.0)
    assert score_metrics(_df(), "worst")["roc_auc"] == pytest.approx(0.0)


def test_bootstrap_diff_positive_when_a_dominates():
    df = pd.concat([_df().assign(cell_idx=lambda d: d.cell_idx + 10 * i) for i in range(20)])
    out = bootstrap_pr_auc_diff(df, "perfect", "worst", n_boot=100)
    assert out["diff_ci_low"] > 0


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(loc=1.5, size=5000)) > PSI_ALERT


def test_psi_counts_missing_share_change():
    a = np.r_[np.zeros(900), np.full(100, np.nan)]
    b = np.r_[np.zeros(500), np.full(500, np.nan)]
    assert psi(a, b) > PSI_ALERT


def test_drift_report_status_counts():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(size=2000), "y": rng.normal(size=2000)})
    cur = pd.DataFrame({"x": rng.normal(size=2000), "y": rng.normal(loc=2, size=2000)})
    rep = drift_report(ref, cur, ["x", "y"])
    assert rep["features"]["x"]["status"] == "ok" and rep["features"]["y"]["status"] == "alert"
    assert rep["n_alert"] == 1
