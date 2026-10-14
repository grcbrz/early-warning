"""Feature engineering: drought metric, spatial lag, panel and split rules."""

import numpy as np
import pandas as pd
import pytest

from congo_ews.features import (
    FEATURE_COLUMNS,
    add_labels,
    annual_climate,
    mcwd_series,
    neighbour_mean,
    split_by_year,
)


def test_mcwd_accumulates_deficit_and_resets_when_wet():
    # ET fixed at 100 mm/month: deficits -50, -50, then a wet month refills.
    p = np.array([[50.0], [50.0], [300.0], [80.0]])
    cwd = mcwd_series(p, et=100.0)
    np.testing.assert_allclose(cwd[:, 0], [-50, -100, 0, -20])


def test_mcwd_propagates_nan_as_missing_month_not_crash():
    p = np.array([[50.0], [np.nan], [50.0]])
    cwd = mcwd_series(p, et=100.0)
    assert cwd[0, 0] == -50 and np.isnan(cwd[1, 0]) and cwd[2, 0] == -100


def test_annual_climate_uses_trailing_baseline_only():
    months = pd.date_range("2001-01-01", periods=12 * 8, freq="MS")
    p = np.full((len(months), 1), 150.0)
    p[months.year == 2008] = 50.0  # drought year at the end
    out = annual_climate(p, months, baseline_years=5)
    row_2007 = out[out.year == 2007].iloc[0]
    row_2008 = out[out.year == 2008].iloc[0]
    assert row_2007["precip_anom"] == pytest.approx(0.0)
    assert row_2008["precip_anom"] == pytest.approx(-1200.0)
    assert row_2008["mcwd_anom"] < 0
    assert out.year.min() == 2006  # needs 5 full previous years


def test_neighbour_mean_excludes_self_and_invalid():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 100.0, 6.0], [7.0, 8.0, 9.0]])
    valid = np.ones_like(a, bool)
    valid[0, 0] = False
    out = neighbour_mean(a, valid)
    assert out[1, 1] == pytest.approx(np.mean([2, 3, 4, 6, 7, 8, 9]))
    assert out[0, 1] == pytest.approx(np.mean([3, 4, 100, 6]))


def test_add_labels_threshold_from_train_years_only():
    df = pd.DataFrame({"year": [2005] * 10 + [2020] * 10,
                       "target_loss": list(np.linspace(0, 0.009, 10)) + [0.5] * 10})
    out, thr = add_labels(df, train_years=range(2005, 2017), quantile=0.9)
    assert thr == pytest.approx(np.quantile(np.linspace(0, 0.009, 10), 0.9))
    assert out.loc[out.year == 2020, "label"].all()


def test_split_by_year_is_disjoint_and_ordered():
    df = pd.DataFrame({"year": range(2005, 2025)})
    s = split_by_year(df)
    assert s["train"].year.max() < s["val"].year.min() <= s["val"].year.max() < s["test"].year.min()
    assert s["test"].year.max() == 2023 and s["live"].year.unique().tolist() == [2024]


def test_feature_columns_have_no_target_or_future_fields():
    assert not any(c.startswith(("target", "label")) for c in FEATURE_COLUMNS)
