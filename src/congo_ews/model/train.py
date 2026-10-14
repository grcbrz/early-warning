"""Candidate models, selection on validation, single test evaluation (SPEC §4.3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from congo_ews.features import FEATURE_COLUMNS, PRESSURE_FEATURES
from congo_ews.model.evaluate import bootstrap_pr_auc_diff, score_metrics

SEED = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    features: list[str]
    make: Callable[[], BaseEstimator] | None  # None = rule-based score, no fitting

    def score(self, model: BaseEstimator | None, df: pd.DataFrame) -> np.ndarray:
        if model is None:
            return df[self.features[0]].to_numpy()
        return model.predict_proba(df[self.features])[:, 1]


def _hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=SEED,
    )


CANDIDATES = [
    Candidate("B0_persistence", ["loss_t"], None),
    Candidate("B1_logreg_pressure", PRESSURE_FEATURES, lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))),
    Candidate("M1_hgb_pressure", PRESSURE_FEATURES, _hgb),
    Candidate("M2_hgb_all", FEATURE_COLUMNS, _hgb),
]


def fit(c: Candidate, df: pd.DataFrame) -> BaseEstimator | None:
    if c.make is None:
        return None
    model = c.make()
    model.fit(df[c.features], df["label"].astype(int))
    return model


def run_experiment(splits: dict[str, pd.DataFrame], n_boot: int = 1000) -> tuple[dict, pd.DataFrame, Candidate, BaseEstimator | None]:
    """Fit on train, select on val, refit on train+val, score test once, refit all for live."""
    train, val, test, live = (splits[k] for k in ("train", "val", "test", "live"))
    results: dict = {"val": {}, "test": {}}
    for c in CANDIDATES:
        m = fit(c, train)
        v = val.assign(score=c.score(m, val))
        results["val"][c.name] = score_metrics(v, "score", is_probability=c.make is not None)

    learned = [c for c in CANDIDATES if c.make is not None]
    best = max(learned, key=lambda c: results["val"][c.name]["pr_auc"])
    results["selected"] = best.name

    trainval = pd.concat([train, val])
    scored_test = test[["year", "cell_idx", "label"]].copy()
    for c in CANDIDATES:
        m = fit(c, trainval)
        scored_test[c.name] = c.score(m, test)
        results["test"][c.name] = score_metrics(scored_test, c.name, is_probability=c.make is not None)
    results["test_bootstrap_vs_persistence"] = bootstrap_pr_auc_diff(
        scored_test, best.name, "B0_persistence", n_boot=n_boot)
    results["test_bootstrap_vs_pressure_only"] = bootstrap_pr_auc_diff(
        scored_test, "M2_hgb_all", "M1_hgb_pressure", n_boot=n_boot)

    # Explain the selected model on validation data (model trained on train only).
    m_train = fit(best, train)
    imp = permutation_importance(m_train, val[best.features], val["label"].astype(int),
                                 scoring="average_precision", n_repeats=10, random_state=SEED)
    results["permutation_importance_val"] = dict(sorted(
        zip(best.features, imp.importances_mean.round(5).tolist()), key=lambda kv: -kv[1]))

    final = fit(best, pd.concat([train, val, test]))
    preds = pd.concat([
        scored_test[["year", "cell_idx", best.name]].rename(columns={best.name: "risk"}).assign(split="test"),
        live[["year", "cell_idx"]].assign(risk=best.score(final, live), split="live"),
    ], ignore_index=True)
    return results, preds, best, final
