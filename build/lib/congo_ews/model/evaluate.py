"""Evaluation metrics (SPEC §4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def at_fraction(df: pd.DataFrame, score_col: str, frac: float = 0.10) -> dict[str, float]:
    """Precision, recall and the best achievable recall when only the top ``frac`` of cells
    (ranked within each year, since capacity is allocated yearly) can be checked; averaged over years.
    """
    prec, rec, ceiling = [], [], []
    for _, g in df.groupby("year"):
        pos = g["label"].sum()
        if pos == 0:
            continue
        k = max(1, int(np.ceil(frac * len(g))))
        hits = g.nlargest(k, score_col)["label"].sum()
        prec.append(hits / k)
        rec.append(hits / pos)
        ceiling.append(min(k, pos) / pos)
    if not rec:
        return {"precision": float("nan"), "recall": float("nan"), "recall_ceiling": float("nan")}
    return {"precision": float(np.mean(prec)), "recall": float(np.mean(rec)),
            "recall_ceiling": float(np.mean(ceiling))}


def recall_at_fraction(df: pd.DataFrame, score_col: str, frac: float = 0.10) -> float:
    return at_fraction(df, score_col, frac)["recall"]


def score_metrics(df: pd.DataFrame, score_col: str, is_probability: bool = True) -> dict[str, float]:
    y, s = df["label"].astype(int).to_numpy(), df[score_col].to_numpy()
    out = {
        "n": int(len(df)),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }
    top = at_fraction(df, score_col, 0.10)
    out.update({"precision_at_10pct": top["precision"], "recall_at_10pct": top["recall"],
                "recall_at_10pct_ceiling": top["recall_ceiling"]})
    if is_probability:
        out["brier"] = float(brier_score_loss(y, s))
    return out


def bootstrap_pr_auc_diff(
    df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Cluster bootstrap over cells: CI for PR-AUC(a) - PR-AUC(b).

    Resampling whole cells (all their years together) respects within-cell correlation.
    """
    rng = np.random.default_rng(seed)
    cells = df["cell_idx"].unique()
    groups = {c: idx for c, idx in df.groupby("cell_idx").indices.items()}
    y = df["label"].astype(int).to_numpy()
    a, b = df[col_a].to_numpy(), df[col_b].to_numpy()
    diffs, pa = [], []
    for _ in range(n_boot):
        pick = rng.choice(cells, size=len(cells), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        if y[idx].sum() == 0:
            continue
        ap_a = average_precision_score(y[idx], a[idx])
        diffs.append(ap_a - average_precision_score(y[idx], b[idx]))
        pa.append(ap_a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pa_lo, pa_hi = np.percentile(pa, [2.5, 97.5])
    return {"diff_mean": float(np.mean(diffs)), "diff_ci_low": float(lo), "diff_ci_high": float(hi),
            "pr_auc_ci_low": float(pa_lo), "pr_auc_ci_high": float(pa_hi)}
