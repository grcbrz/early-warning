"""Cell-year feature panel and labels (SPEC §4.2). Features at year t use data <= Dec t."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from congo_ews.config import MIN_FOREST_FRACTION, Grid
from congo_ews.process.indicators import kendall_tau

PRESSURE_FEATURES = [
    "loss_t", "loss_t1", "loss_t2", "cum_loss", "forest_remaining", "neighbour_loss",
]
RESILIENCE_FEATURES = ["ar1", "variance", "ar1_tau", "ndvi_anom", "straddles_break"]
CLIMATE_FEATURES = ["precip_anom", "mcwd_anom"]
FEATURE_COLUMNS = PRESSURE_FEATURES + RESILIENCE_FEATURES + CLIMATE_FEATURES

SPLITS = {
    "train": range(2005, 2017),
    "val": range(2017, 2020),
    "test": range(2020, 2024),
    "live": range(2024, 2025),
}


def mcwd_series(precip: np.ndarray, et: float = 100.0) -> np.ndarray:
    """Running cumulative water deficit (Aragao et al. 2007), carried across years.

    CWD_n = min(0, CWD_{n-1} + P_n - ET). A missing month yields NaN and leaves the state as is.
    """
    state = np.zeros(precip.shape[1])
    out = np.full(precip.shape, np.nan)
    for i, p in enumerate(precip):
        ok = np.isfinite(p)
        state[ok] = np.minimum(0.0, state[ok] + p[ok] - et)
        out[i, ok] = state[ok]
    return out


def annual_climate(precip: np.ndarray, months: pd.DatetimeIndex, baseline_years: int = 5) -> pd.DataFrame:
    """Annual precipitation and MCWD anomalies against the trailing ``baseline_years`` years."""
    years = months.year.to_numpy()
    cwd = mcwd_series(precip)
    uniq = np.unique(years)
    with np.errstate(all="ignore"):
        total = np.stack([np.nansum(precip[years == y], axis=0) for y in uniq])
        mcwd = np.stack([np.nanmin(cwd[years == y], axis=0) for y in uniq])
    frames = []
    for i in range(baseline_years, len(uniq)):
        prev = slice(i - baseline_years, i)
        frames.append(pd.DataFrame({
            "year": uniq[i],
            "cell_idx": np.arange(precip.shape[1]),
            "precip_anom": total[i] - total[prev].mean(axis=0),
            "mcwd_anom": mcwd[i] - mcwd[prev].mean(axis=0),
        }))
    return pd.concat(frames, ignore_index=True)


def neighbour_mean(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean of the valid 8-neighbours of each cell (queen contiguity), excluding the cell itself."""
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    vals = np.where(valid, a, 0.0)
    s = convolve(vals, kernel, mode="constant", cval=0.0)
    n = convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
    with np.errstate(all="ignore"):
        return np.where(n > 0, s / n, np.nan)


def build_panel(
    forest_frac_2000: np.ndarray,
    loss_frac: np.ndarray,
    loss_years: np.ndarray,
    indicators: pd.DataFrame,
    climate: pd.DataFrame,
    grid: Grid,
) -> pd.DataFrame:
    """Assemble the cell-year table. ``loss_frac`` is (year, row, col) relative to 2000 forest."""
    in_scope = forest_frac_2000 >= MIN_FOREST_FRACTION
    rows, cols = np.where(in_scope)
    cell_idx = rows * grid.n_cols + cols
    has_forest = forest_frac_2000 > 0
    cum = np.cumsum(loss_frac, axis=0)
    frames = []
    for k, year in enumerate(loss_years):
        if k < 2:
            continue
        nb = neighbour_mean(loss_frac[k], has_forest)
        frames.append(pd.DataFrame({
            "year": year,
            "cell_idx": cell_idx,
            "row": rows,
            "col": cols,
            "lat": grid.lat_centers[rows],
            "lon": grid.lon_centers[cols],
            "loss_t": loss_frac[k][in_scope],
            "loss_t1": loss_frac[k - 1][in_scope],
            "loss_t2": loss_frac[k - 2][in_scope],
            "cum_loss": cum[k][in_scope],
            "forest_remaining": (forest_frac_2000 * (1 - cum[k]))[in_scope],
            "neighbour_loss": nb[in_scope],
            "target_loss": loss_frac[k + 1][in_scope] if k + 1 < len(loss_years) else np.nan,
        }))
    panel = pd.concat(frames, ignore_index=True)

    ind = indicators.sort_values(["cell_idx", "year"]).copy()
    ind["ar1_tau"] = (ind.groupby("cell_idx")["ar1"]
                      .transform(lambda s: s.rolling(5, min_periods=5).apply(kendall_tau, raw=True)))
    ind["straddles_break"] = ind["straddles_break"].astype(float)
    panel = panel.merge(ind[["year", "cell_idx", *RESILIENCE_FEATURES]], on=["year", "cell_idx"], how="left")
    panel = panel.merge(climate, on=["year", "cell_idx"], how="left")
    return panel.sort_values(["year", "cell_idx"]).reset_index(drop=True)


def add_labels(df: pd.DataFrame, train_years: Iterable[int], quantile: float = 0.9) -> tuple[pd.DataFrame, float]:
    """Binary target: next-year loss >= the ``quantile`` of training-year next-year loss."""
    train_mask = df["year"].isin(list(train_years)) & df["target_loss"].notna()
    threshold = float(np.quantile(df.loc[train_mask, "target_loss"], quantile))
    out = df.copy()
    out["label"] = np.where(out["target_loss"].notna(), out["target_loss"] >= threshold, np.nan)
    return out, threshold


def split_by_year(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: df[df["year"].isin(list(years))] for name, years in SPLITS.items()}
