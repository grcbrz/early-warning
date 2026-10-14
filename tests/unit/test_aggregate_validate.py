"""Grid aggregation (NDVI, Hansen) and data-validation rules (AC2)."""

import numpy as np
import pandas as pd
import pytest

from congo_ews.process.aggregate import block_mean, hansen_strip_counts
from congo_ews.process.validate import DataValidationError, validate_monthly_stack


def test_block_mean_respects_min_valid_fraction():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    a[0, 0] = a[0, 1] = a[1, 0] = np.nan  # 3/4 missing in the top-left block
    out = block_mean(a, factor=2, min_valid_frac=0.5)
    assert out.shape == (2, 2)
    assert np.isnan(out[0, 0])
    assert out[1, 1] == pytest.approx(np.mean([10, 11, 14, 15]))


def test_block_mean_rejects_non_divisible_shape():
    with pytest.raises(ValueError):
        block_mean(np.zeros((5, 4)), factor=2)


def test_hansen_strip_counts_forest_and_loss_by_year():
    # 2 x 4 pixel strip, block of 2 -> two cells.
    tc = np.array([[80, 10, 50, 50], [90, 90, 50, 0]], dtype=np.uint8)
    ly = np.array([[3, 3, 0, 1], [0, 24, 1, 1]], dtype=np.uint8)
    forest, loss = hansen_strip_counts(tc, ly, block=2, canopy_threshold=30, n_years=24)
    np.testing.assert_array_equal(forest, [3, 3])
    # cell 0: forest pixels (80,ly3), (90,ly0), (90,ly24) -> loss in 2003 and 2024 only
    assert loss[0, 2] == 1 and loss[0, 23] == 1 and loss[0].sum() == 2
    # cell 1: forest pixels (50,ly0),(50,ly1),(50,ly1); the 0-canopy pixel with loss is ignored
    assert loss[1, 0] == 2 and loss[1].sum() == 2


def _good_stack(n: int = 24) -> tuple[np.ndarray, pd.DatetimeIndex]:
    months = pd.date_range("2001-01-01", periods=n, freq="MS")
    return np.full((n, 4, 6), 0.7, dtype=np.float32), months


def test_validate_accepts_good_stack():
    stack, months = _good_stack()
    report = validate_monthly_stack(stack, months, expected_shape=(4, 6), value_range=(-0.2, 1.0))
    assert report["n_months"] == 24 and report["missing_months"] == []


def test_validate_rejects_out_of_range():
    stack, months = _good_stack()
    stack[3, 0, 0] = 1.7
    with pytest.raises(DataValidationError, match="range"):
        validate_monthly_stack(stack, months, expected_shape=(4, 6), value_range=(-0.2, 1.0))


def test_validate_rejects_wrong_shape():
    stack, months = _good_stack()
    with pytest.raises(DataValidationError, match="shape"):
        validate_monthly_stack(stack, months, expected_shape=(5, 6), value_range=(-0.2, 1.0))


def test_validate_reports_missing_months_and_fails_over_budget():
    stack, months = _good_stack()
    keep = np.ones(24, bool)
    keep[[5, 6]] = False
    report = validate_monthly_stack(stack[keep], months[keep], expected_shape=(4, 6),
                                    value_range=(-0.2, 1.0), max_missing_months=2)
    assert report["missing_months"] == ["2001-06", "2001-07"]
    with pytest.raises(DataValidationError, match="missing"):
        validate_monthly_stack(stack[keep], months[keep], expected_shape=(4, 6),
                               value_range=(-0.2, 1.0), max_missing_months=1)
