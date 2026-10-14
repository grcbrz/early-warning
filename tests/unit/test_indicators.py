"""AC3: resilience indicators recover known dynamics; deseasonalisation is era-aware."""

import numpy as np
import pandas as pd
import pytest

from congo_ews.process.indicators import (
    ar1_columns,
    deseasonalize,
    detrend_columns,
    kendall_tau,
    rolling_indicators,
)


def simulate_ar1(phi: float, n: int, n_series: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((n, n_series))
    eps = rng.standard_normal((n, n_series))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_ar1_recovers_phi_long_series():
    rng = np.random.default_rng(0)
    x = simulate_ar1(0.6, 2000, 3, rng)
    np.testing.assert_allclose(ar1_columns(x), 0.6, atol=0.05)


def test_ar1_ranks_phi_correctly_for_short_windows():
    rng = np.random.default_rng(1)
    low = ar1_columns(simulate_ar1(0.2, 60, 500, rng))
    high = ar1_columns(simulate_ar1(0.8, 60, 500, rng))
    assert np.median(high) > np.median(low) + 0.4


def test_ar1_ignores_missing_pairs():
    x = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
    assert np.isfinite(ar1_columns(x, min_valid=4)[0])


def test_ar1_nan_when_too_few_valid():
    x = np.array([[1.0], [np.nan], [np.nan], [4.0]])
    assert np.isnan(ar1_columns(x, min_valid=3)[0])


def test_ar1_constant_series_is_nan_not_error():
    assert np.isnan(ar1_columns(np.ones((50, 1)))[0])


def test_detrend_removes_linear_trend_with_gaps():
    t = np.arange(100, dtype=float)
    x = np.column_stack([3 + 0.5 * t, -2 * t])
    x[10, 0] = np.nan
    d = detrend_columns(x)
    assert np.nanmax(np.abs(d)) < 1e-9
    assert np.isnan(d[10, 0])


def test_deseasonalize_removes_cycle_and_sensor_offset():
    months = pd.date_range("2001-01-01", periods=240, freq="MS")
    era = np.where(months.year < 2011, "AVHRR", "VIIRS")
    seasonal = np.sin(2 * np.pi * months.month.to_numpy() / 12)
    offset = np.where(era == "VIIRS", 0.15, 0.0)  # calibration step at the sensor switch
    rng = np.random.default_rng(2)
    x = (0.7 + 0.1 * seasonal + offset)[:, None] + 0.01 * rng.standard_normal((240, 2))
    z = deseasonalize(x, months.month.to_numpy(), era)
    for e in ("AVHRR", "VIIRS"):
        block = z[era == e]
        assert abs(np.nanmean(block)) < 1e-6
        assert np.nanstd(block) == pytest.approx(1.0, abs=0.05)
    # The step must not survive into the residuals.
    assert abs(np.nanmean(z[era == "VIIRS"]) - np.nanmean(z[era == "AVHRR"])) < 1e-6


def test_rolling_indicators_are_causal_and_flag_break():
    months = pd.date_range("2001-01-01", periods=12 * 10, freq="MS")
    rng = np.random.default_rng(3)
    z = rng.standard_normal((len(months), 4))
    era = np.where(months.year < 2006, "AVHRR", "VIIRS")
    out = rolling_indicators(z, months, era, window=60, min_valid=48)
    assert set(out.columns) >= {"year", "cell_idx", "ar1", "variance", "straddles_break"}
    first_year = out["year"].min()
    assert first_year == 2005  # first full 60-month window ends Dec 2005
    # Changing data after Dec 2007 must not change indicators for 2007.
    z2 = z.copy()
    z2[months > "2007-12-31"] = 99.0
    out2 = rolling_indicators(z2, months, era, window=60, min_valid=48)
    a = out[out.year == 2007].sort_values("cell_idx")["ar1"].to_numpy()
    b = out2[out2.year == 2007].sort_values("cell_idx")["ar1"].to_numpy()
    np.testing.assert_array_equal(a, b)
    assert out.loc[out.year == 2007, "straddles_break"].all()
    assert not out.loc[out.year == 2005, "straddles_break"].any()


def test_kendall_tau_monotone():
    assert kendall_tau(np.array([0.1, 0.2, 0.3, 0.4, 0.5])) == pytest.approx(1.0)
    assert kendall_tau(np.array([0.5, 0.4, 0.3, 0.2, 0.1])) == pytest.approx(-1.0)
    assert np.isnan(kendall_tau(np.array([0.1, np.nan, np.nan, np.nan, 0.2])))


def test_rolling_variance_reflects_amplitude_not_standardised():
    months = pd.date_range("2001-01-01", periods=72, freq="MS")
    rng = np.random.default_rng(4)
    x = np.column_stack([0.01 * rng.standard_normal(72), 0.05 * rng.standard_normal(72)])
    out = rolling_indicators(x, months, np.array(["AVHRR"] * 72), window=60, min_valid=40)
    v = out[out.year == 2005].sort_values("cell_idx")["variance"].to_numpy()
    assert v[1] > 10 * v[0]
