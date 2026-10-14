# Monitoring and retraining plan

The system runs on an annual cycle, because its labels (Hansen GFC loss) are released once a year.

## Cadence

| When | Trigger | Action |
|---|---|---|
| Monthly | New NDVI and PERSIANN month on AWS | `scripts/fetch.py ndvi|precip`; validation report must pass (no gaps, range, grid shape) |
| Yearly, on Hansen GFC release (usually Q2) | New loss year `t` | Rebuild panel → score year `t` features → publish risk for `t+1` |
| Same run | Labels for year `t` now known | Score last year's live forecast against realised loss (**delayed-label evaluation**) |
| Same run | Retraining | Refit selected model on all labelled years; re-run selection only if the gates below fail |

## Checks and gates

1. **Data contract** (`process/validate.py`): fail the run on out-of-range values, wrong grid, or missing months.
2. **Calibration era** (`process/build.sensor_era`): a new satellite platform in the NDVI file names (e.g. S-NPP →
   NOAA-20 in Jan 2023) starts a new era automatically. Windows that straddle a change are flagged
   (`straddles_break`), and short segments are dropped.
3. **Input drift** (`monitoring/drift.py`, `artifacts/drift_report.json`): PSI per feature for the new year,
   against both the training years and the three most recent labelled years.
   - PSI ≥ 0.10 → warn, ≥ 0.25 → alert.
   - The actionable reference is **recent years**: loss features trend upward, so every year alerts against
     2005–2016 by construction.
   - An alert on a feature the selected model uses blocks automatic publication until reviewed. Current state
     (2024): all model inputs OK vs recent; NDVI variance/anomaly alert, traced to the NOAA-20 switch.
4. **Performance** (delayed labels): PR-AUC and precision@10 % of last year's forecast. Retrain-and-reselect if
   PR-AUC falls below persistence's for the same year, or precision@10 % drops by more than 0.10 from the test
   average (0.95).
5. **Label shift**: prevalence of the high-loss label is ~19–21 % since 2013 against ~10 % in training. The fixed
   threshold keeps the target's meaning stable; recalibrate probabilities (isotonic on the last 3 labelled years)
   if Brier score rises above 0.08.

## Operational notes
- Serving is stateless: the image bundles `artifacts/` and `data/processed/`; a new year is a new image tag.
- `/plot-check` reads Hansen tiles over HTTP range requests (row-striped GeoTIFFs, ~0.1 s per 4 km plot). For
  volume, mount local tiles via `HANSEN_DIR`.
- Log request count, latency p95 and 4xx/5xx rates per endpoint; 422 rates on `/plot-check` show client misuse.
