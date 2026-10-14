# Model card: next-year high-loss risk (M1_hgb_pressure)

**Task.** For each 0.25° Congo Basin forest cell and year `t`, probability that tree-cover loss in `t+1` is at
least 0.867 % of the cell's 2000 forest (90th percentile of training years).

**Model.** scikit-learn `HistGradientBoostingClassifier` (300 iterations, lr 0.05, 15 leaves, min 50 samples
per leaf, L2 1.0, seed 42) on six loss-history features: loss at `t`, `t−1`, `t−2`, cumulative loss,
remaining forest, 8-neighbour loss. Selected on 2017–19 validation PR-AUC over a persistence baseline, a
logistic regression and the same booster with resilience and climate features.

**Data.** 4,623 cells × feature years 2005–2023 (labelled), Hansen GFC v1.12, NOAA CDR NDVI, PERSIANN-CDR.

## Results (test: feature years 2020–23, targets 2021–24; scored once)

| Model | PR-AUC | ROC-AUC | Precision@10 % | Recall@10 % (ceiling 0.51) | Brier |
|---|---|---|---|---|---|
| Persistence | 0.847 | 0.957 | 0.91 | 0.46 | – |
| Logistic regression | 0.872 | 0.963 | 0.92 | 0.47 | 0.064 |
| **Boosting, loss history (selected)** | **0.888** | **0.966** | **0.95** | **0.48** | **0.059** |
| Boosting, + resilience & climate | 0.878 | 0.959 | 0.95 | 0.48 | 0.061 |

- Gain over persistence: +0.041 PR-AUC, 95 % cluster-bootstrap CI [0.032, 0.051].
- Resilience + climate features: −0.010 PR-AUC, CI [−0.013, −0.007]. They do not help a one-year forecast at this
  resolution. Spearman correlation between AR1 and next-year loss is |ρ| < 0.07 in every year checked.
- Permutation importance (val): cumulative loss dominates (0.50), then recent loss (0.04 each).

## Intended use
Ranking landscapes for monitoring effort and for context in supplier due diligence. **Not** for plot-level
compliance decisions, and not a tipping-point estimate.

## Known weaknesses
- High scores concentrate where loss is already high; the model finds continuing frontiers, not new ones.
  Evaluating "first high-loss year" cells separately is the obvious next analysis.
- Hansen detection improved after 2011, inflating later prevalence (3–5 % → ~20 %).
- No roads, settlements, concessions or fire inputs; these are the expected next gains.
