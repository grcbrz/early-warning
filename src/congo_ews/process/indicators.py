"""Critical-slowing-down indicators (Dakos et al. 2012; Boulton, Lenton & Boers 2022).

Arrays are (time, cell): rows are consecutive months, columns are grid cells. All functions
tolerate NaN (cloud gaps) and are vectorised across cells.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def deseasonalize(x: np.ndarray, month_of_year: np.ndarray, era: np.ndarray,
                  standardize: bool = True) -> np.ndarray:
    """Remove the month-of-year climatology (and optionally standardise), separately per sensor era.

    Doing it per era absorbs calibration steps (e.g. AVHRR -> VIIRS in 2014) that would
    otherwise show up as a spurious trend or variance jump.
    """
    if not (len(x) == len(month_of_year) == len(era)):
        raise ValueError("x, month_of_year and era must have the same length")
    z = np.full(x.shape, np.nan, dtype=np.float64)
    for e in np.unique(era):
        in_era = era == e
        resid = np.full((in_era.sum(), x.shape[1]), np.nan)
        block, moy = x[in_era].astype(np.float64), month_of_year[in_era]
        for m in np.unique(moy):
            rows = moy == m
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                resid[rows] = block[rows] - np.nanmean(block[rows], axis=0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd = np.nanstd(resid, axis=0)
        sd[sd == 0] = np.nan
        z[in_era] = resid / sd if standardize else resid
    return z


def detrend_columns(x: np.ndarray) -> np.ndarray:
    """Subtract a per-column least-squares line fitted on the valid (non-NaN) rows."""
    t = np.arange(x.shape[0], dtype=np.float64)[:, None]
    valid = np.isfinite(x)
    n = valid.sum(axis=0)
    with np.errstate(all="ignore"):
        t_mean = np.where(valid, t, 0).sum(axis=0) / n
        x_mean = np.where(valid, x, 0).sum(axis=0) / n
        dt = np.where(valid, t - t_mean, 0)
        dx = np.where(valid, x - x_mean, 0)
        slope = (dt * dx).sum(axis=0) / (dt**2).sum(axis=0)
    slope = np.where(np.isfinite(slope), slope, 0.0)
    return x - (x_mean + slope * (t - t_mean))


def ar1_columns(x: np.ndarray, min_valid: int = 10) -> np.ndarray:
    """Lag-1 autocorrelation per column, using only pairs where both months are valid."""
    a, b = x[:-1], x[1:]
    pair = np.isfinite(a) & np.isfinite(b)
    n_pairs = pair.sum(axis=0)
    with np.errstate(all="ignore"):
        a_mean = np.where(pair, a, 0).sum(axis=0) / n_pairs
        b_mean = np.where(pair, b, 0).sum(axis=0) / n_pairs
        da = np.where(pair, a - a_mean, 0)
        db = np.where(pair, b - b_mean, 0)
        r = (da * db).sum(axis=0) / np.sqrt((da**2).sum(axis=0) * (db**2).sum(axis=0))
    enough = (np.isfinite(x).sum(axis=0) >= min_valid) & (n_pairs >= 3)
    return np.where(enough & np.isfinite(r), r, np.nan)


def rolling_indicators(
    x: np.ndarray,
    months: pd.DatetimeIndex,
    era: np.ndarray,
    window: int = 60,
    min_valid: int = 40,
    min_segment: int = 24,
) -> pd.DataFrame:
    """AR1, variance and NDVI anomaly in trailing ``window``-month windows ending each December.

    Everything, including the seasonal climatology, is estimated inside the window, so only
    data up to December of the reported year is used (causal by construction). Within a
    window each sensor era is deseasonalised separately; an era segment shorter than
    ``min_segment`` months is dropped because its climatology cannot be estimated.
    """
    expected = pd.date_range(months[0], periods=len(months), freq="MS")
    if not months.equals(expected):
        raise ValueError("months must be a gap-free monthly index (fill gaps with NaN rows)")
    moy = months.month.to_numpy()
    frames = []
    for end in np.where(moy == 12)[0]:
        start = end - window + 1
        if start < 0:
            continue
        sl = slice(start, end + 1)
        era_w = era[sl]
        z = deseasonalize(x[sl], moy[sl], era_w)
        for e in np.unique(era_w):
            if (era_w == e).sum() < min_segment:
                z[era_w == e] = np.nan
        w = detrend_columns(z)
        n_valid = np.isfinite(w).sum(axis=0)
        ar1 = ar1_columns(w, min_valid=min_valid)
        # Variance needs unstandardised residuals; use the latest era segment only, because
        # NDVI amplitude is not comparable across sensors.
        last = era_w == era_w[-1]
        raw = detrend_columns(deseasonalize(x[sl][last], moy[sl][last], era_w[last], standardize=False))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells -> NaN is intended
            var = np.nanvar(raw, axis=0, ddof=1) if last.sum() >= min_segment else np.full(x.shape[1], np.nan)
            anom = np.nanmean(z[-12:], axis=0)
        ok = n_valid >= min_valid
        frames.append(pd.DataFrame({
            "year": months[end].year,
            "cell_idx": np.arange(x.shape[1]),
            "ar1": ar1,
            "variance": np.where(ok, var, np.nan),
            "ndvi_anom": np.where(ok, anom, np.nan),
            "n_valid": n_valid,
            "straddles_break": len(np.unique(era_w)) > 1,
        }))
    return pd.concat(frames, ignore_index=True)


def kendall_tau(series: np.ndarray, min_valid: int = 4) -> float:
    """Kendall rank correlation of ``series`` against time; NaN if too few points."""
    valid = np.isfinite(series)
    if valid.sum() < min_valid:
        return float("nan")
    tau, _ = kendalltau(np.arange(len(series))[valid], series[valid])
    return float(tau)
