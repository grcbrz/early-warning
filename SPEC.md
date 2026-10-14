# SPEC: Congo Basin forest-resilience early-warning system (congo-ews)

Status: v0.1, written before modelling. Changes after test-set evaluation are logged in §9.

## 1. Problem

The Congo Basin holds the world's second-largest tropical rainforest. Nobre and colleagues showed the Amazon can tip
to degraded savanna once deforestation and warming pass a threshold (Lovejoy & Nobre 2018, *Sci. Adv.*), and
Boulton, Lenton & Boers (2022, *Nat. Clim. Change*) detected the precursor: **rising lag-1 autocorrelation (AR1)
of vegetation indices**, i.e. critical slowing down, across >75 % of the Amazon. **No equivalent threshold has been
published for the Congo**, so this project does not import the Amazon's 20–25 % figure.

Decision-makers (CAFI/national REDD+ programmes, NGOs planning patrols, companies doing EU Deforestation Regulation
(EUDR, Reg. 2023/1115) due diligence) need two things per ~28 km grid cell, refreshed yearly:

1. **Resilience signal**: is the forest's recovery from short-term perturbations slowing (AR1 up, variance up)?
2. **Forest-loss risk**: how likely is a high-loss year next year, so limited field capacity goes to the right cells?

## 2. Users and decisions

| User | Decision supported | Output |
|---|---|---|
| Conservation / REDD+ planner | Where to send limited monitoring capacity next year | Ranked cell risk, top-k list |
| Researcher | Where resilience is declining, and whether it adds forecasting skill | AR1, variance, Kendall τ maps; ablation |
| EUDR due-diligence analyst | Screen a supplier plot for post-2020 tree-cover loss and surrounding risk | `/plot-check` API |

Non-goals: predicting a basin-wide tipping point; legal EUDR compliance determination (Hansen "loss" includes
natural disturbance and is not the EUDR definition of deforestation); sub-cell (30 m) forecasting.

## 3. Data

| Source | Variable | Resolution | Period | Access |
|---|---|---|---|---|
| NOAA CDR NDVI v5 (AVHRR) / v1 (VIIRS) | NDVI | 0.05°, daily | 2001–2025 | AWS Open Data `noaa-cdr-ndvi-pds`, HTTP range reads |
| PERSIANN-CDR v1r1 | Precipitation | 0.25°, daily | 2001–2025 | AWS Open Data `noaa-cdr-precip-persiann-pds` |
| Hansen/UMD Global Forest Change v1.12 | Tree cover 2000, loss year | 30 m | 2001–2024 | Google Cloud Storage |

Region: lat −10…6, lon 8…32 (WGS84). Analysis grid: 0.25° (64 × 96 cells). A cell is in scope if ≥ 50 % of its
area had ≥ 30 % canopy cover in 2000.

Known data issues, handled explicitly:
- **Sensor breaks** Jan 2014 (AVHRR → VIIRS S-NPP) and Jan 2023 (S-NPP → NOAA-20, found during validation):
  different calibration. Residuals are deseasonalised and standardised **per era**; windows that straddle a
  break are flagged.
- **Persistent cloud** over the basin: daily NDVI composited monthly with maximum-value compositing (8 sampled
  days/month); cell-months with < 50 % valid pixels are set missing.
- **NDVI saturation** over dense canopy: limits sensitivity; documented as a limitation (VOD is the upgrade path).
- **Fire (NASA FIRMS)**: blocked from the build environment; out of scope for v0.1.

## 4. Method

### 4.1 Resilience indicators (per cell, monthly series)
1. Monthly NDVI cell mean → inside each window, deseasonalise by subtracting the month-of-year climatology of
   each sensor era present, then z-score within the era (window-local, so no future data is used).
2. For each year-end `t`, take the trailing 60-month window ending Dec `t`, remove a linear trend **inside the
   window**, compute AR1 (lag-1 Pearson autocorrelation) and variance. Require ≥ 40 valid months; inside a window,
   a sensor-era segment shorter than 24 months is dropped (its climatology cannot be estimated).
3. `ar1_tau`: Kendall τ of the AR1 series over the trailing 5 year-ends (trend in AR1).

All windows are trailing, so indicators are causal and usable operationally.

### 4.2 Forecasting task
- Unit: cell-year. Features at year `t` use only data up to 31 Dec `t`.
- Target: `y(t+1) = 1` if tree-cover loss in year `t+1` ≥ **τ_loss** of the cell's 2000 forest area. τ_loss is
  fixed from the training years only, as the 90th percentile of annual loss fraction (so ~10 % positives in train).
- Features: loss fraction at `t`, `t−1`, `t−2`; cumulative loss since 2001; remaining forest fraction; spatial lag
  (mean loss fraction of the 8 neighbours at `t`); annual NDVI anomaly; AR1, variance, ar1_tau, straddles-break
  flag; annual precipitation anomaly and MCWD anomaly (Aragão et al. 2007; baseline = training years).
- Temporal split by feature year: train 2005–2016, validation 2017–2019, test 2020–2023 (targets 2021–2024).
  Live forecast: features 2024 → risk for 2025.

### 4.3 Models
| Id | Model | Purpose |
|---|---|---|
| B0 | Persistence: score = loss fraction at `t` | Floor any model must beat |
| B1 | Logistic regression, pressure features only | Simple, interpretable baseline |
| M1 | HistGradientBoosting, pressure features only | Ablation: no resilience/climate |
| M2 | HistGradientBoosting, all features | Candidate |

Selection on validation PR-AUC. Test set is scored once, after selection.

### 4.4 Metrics
Primary: **PR-AUC** (positives are rare). Secondary: ROC-AUC, Brier score, **recall@10 %** (share of next-year
high-loss cells captured if capacity covers the top 10 % of cells). Uncertainty: 1 000-sample bootstrap over cells.

## 5. Interfaces

- Batch pipeline: `make data`, `make features`, `make train`, `make all`.
- REST API (FastAPI):
  - `GET /health`
  - `GET /cells/{cell_id}` → indicators + risk history for one cell
  - `GET /risk?year=&top_k=` → ranked cells
  - `POST /plot-check` (GeoJSON Polygon) → post-2020 Hansen loss inside the plot (30 m), plus risk of the
    containing cells. Screening only, not a compliance verdict.
- Static dashboard (HTML) of risk and AR1 maps and model evaluation.

## 6. Acceptance criteria

| Id | Criterion | Verified by |
|---|---|---|
| AC1 | Pipeline is deterministic: same inputs → identical feature table hash and metrics | `tests/acceptance` |
| AC2 | Validation rejects out-of-range NDVI, wrong grid shape, and month gaps; validation report is written | unit + acceptance |
| AC3 | AR1 estimator recovers φ of a synthetic AR(1) within ±0.05 (n = 2 000) and ranks φ correctly for n = 60 | unit |
| AC4 | No temporal leakage: features at `t` are unchanged when all data after Dec `t` is removed | integration |
| AC5 | All models scored on val and test; selected model's val PR-AUC ≥ persistence; results in `metrics.json` | acceptance |
| AC6 | API contract: valid requests 200 with schema; invalid geometry → 422; plot outside region → 422; unknown cell → 404 | integration |
| AC7 | Drift report (PSI per feature, live year vs training) is produced with documented thresholds | acceptance |

## 7. Engineering standards
Python ≥ 3.11, type hints, modules split by concern (`acquire`, `process`, `model`, `serve`, `monitoring`),
pytest (unit / integration / acceptance), pinned `requirements.txt`, Dockerfile. Network-dependent tests are
marked `network` and skipped by default.

## 8. Limitations accepted up front
0.25° cells average over heterogeneous landscapes; Hansen loss ≠ deforestation; AR1 from NDVI is noisy where
NDVI saturates; one sensor break inside the record; 20-year record gives few independent 5-year windows.

## 9. Change log
- v0.1: initial spec.
- v0.2 (before test scoring): min valid months 48 → 40, because a 10 % cloud-gap rate made every window next to
  a sensor break fail; variance computed on unstandardised residuals of the latest era (z-scoring made it ≡ 1);
  S-NPP → NOAA-20 platform change added as an era; drift also reported against the 3 most recent labelled years.
- v0.2 results: M1 (pressure-only boosting) selected on validation. Test: PR-AUC 0.888 vs persistence 0.847.
  Resilience/climate features reduce PR-AUC by 0.010 (CI excludes 0); hypothesis that they add one-year skill
  is rejected at 0.25°.
