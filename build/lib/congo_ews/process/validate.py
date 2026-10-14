"""Fail-fast data validation for gridded monthly inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input violates the data contract in SPEC.md §3."""


def validate_monthly_stack(
    stack: np.ndarray,
    months: pd.DatetimeIndex,
    expected_shape: tuple[int, int],
    value_range: tuple[float, float],
    max_missing_months: int = 0,
) -> dict:
    """Check grid shape, value range and month continuity; return a report dict."""
    if stack.ndim != 3 or stack.shape[1:] != expected_shape:
        raise DataValidationError(f"shape {stack.shape[1:]} != expected {expected_shape}")
    if len(stack) != len(months):
        raise DataValidationError("stack and months length differ")
    lo, hi = value_range
    finite = stack[np.isfinite(stack)]
    n_bad = int(((finite < lo) | (finite > hi)).sum())
    if n_bad:
        raise DataValidationError(f"{n_bad} values outside range [{lo}, {hi}]")
    full = pd.date_range(months.min(), months.max(), freq="MS")
    missing = [m.strftime("%Y-%m") for m in full.difference(months)]
    if len(missing) > max_missing_months:
        raise DataValidationError(f"{len(missing)} missing months: {missing[:6]}")
    valid_share = np.isfinite(stack).mean(axis=(1, 2))
    return {
        "n_months": int(len(months)),
        "missing_months": missing,
        "valid_share_min": float(valid_share.min()),
        "valid_share_mean": float(valid_share.mean()),
        "value_min": float(finite.min()) if finite.size else None,
        "value_max": float(finite.max()) if finite.size else None,
    }
