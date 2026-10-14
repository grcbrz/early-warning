"""Population Stability Index for feature drift between training and a new scoring year."""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_WARN = 0.10  # moderate shift: investigate
PSI_ALERT = 0.25  # major shift: block automatic scoring, retrain/review


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI with quantile bins from ``expected``; NaNs form their own bin."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    e_fin, a_fin = expected[np.isfinite(expected)], actual[np.isfinite(actual)]
    if e_fin.size == 0 or a_fin.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(e_fin, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(x_all: np.ndarray, x_fin: np.ndarray) -> np.ndarray:
        counts = np.histogram(x_fin, bins=edges)[0]
        counts = np.append(counts, (~np.isfinite(x_all)).sum())
        return np.clip(counts / len(x_all), eps, None)

    e, a = shares(expected, e_fin), shares(actual, a_fin)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> dict:
    rows = {}
    for f in features:
        v = psi(reference[f].to_numpy(), current[f].to_numpy())
        status = "alert" if v >= PSI_ALERT else "warn" if v >= PSI_WARN else "ok"
        rows[f] = {"psi": round(v, 4), "status": status}
    return {
        "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "features": rows,
        "n_alert": sum(r["status"] == "alert" for r in rows.values()),
        "n_warn": sum(r["status"] == "warn" for r in rows.values()),
    }
