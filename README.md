# congo-ews: Congo Basin forest early-warning system

Forecasts which ~28 km forest cells in the Congo Basin are likely to have a high tree-cover-loss year next year,
tracks a critical-slowing-down resilience indicator (NDVI lag-1 autocorrelation, after Boulton et al. 2022), and
screens individual plots for post-2020 loss in the style of an EU Deforestation Regulation (EUDR) check.

Spec: [SPEC.md](SPEC.md) · Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) ·
Monitoring: [docs/MONITORING.md](docs/MONITORING.md) · Dashboard: `dashboard/index.html`

## Results in one paragraph
On 2020–23 (never used for fitting or selection), gradient boosting on loss history reaches PR-AUC 0.888 vs
0.847 for last-year persistence (+0.041, 95 % CI 0.032–0.051); 95 % of the top-10 % cells it flags do have a
high-loss year. Adding NDVI resilience indicators and rainfall/drought anomalies **lowers** PR-AUC by 0.010
(CI −0.013 to −0.007): at 0.25° with NDVI, the resilience signal is too weak to forecast one-year loss, so it is
kept as a separate monitoring layer. Mean annual loss per forest cell rose from 0.16 % (2001–03) to 0.52 %
(2022–24).

## Pipeline

```
acquire/   NOAA CDR NDVI (AWS, HTTP range reads of 60 MB files) · PERSIANN-CDR (AWS) · Hansen GFC v1.12 (GCS)
process/   max-value monthly composites → 0.25° grid → validation report → causal rolling indicators
features   cell-year panel: loss history, spatial lag, AR1/variance/τ, rainfall & MCWD anomalies
model/     persistence · logistic · HGB (pressure) · HGB (all); select on val, test once, cluster bootstrap
monitoring PSI drift vs training and vs recent years
serve/     FastAPI: /health · /risk · /cells/{id} · /plot-check (GeoJSON → 30 m post-2020 loss)
```

Leakage control: every window and anomaly baseline is trailing, and a test removes all data after Dec `t`
and asserts features at `t` are unchanged.

## Run

```bash
pip install -r requirements-dev.txt && pip install -e .
make data          # ~4 GB Hansen tiles + ~10 min of NDVI/rain range reads
make all           # interim → features → train → dashboard
make test          # 51 unit + integration tests, offline
make test-acceptance   # spec AC1–AC7 against built artifacts
make serve         # or: make docker
```

```bash
curl -X POST localhost:8000/plot-check -H 'content-type: application/json' -d \
 '{"plot_id":"demo","geometry":{"type":"Polygon","coordinates":[[[27.36,2.86],[27.39,2.86],[27.39,2.89],[27.36,2.89],[27.36,2.86]]]}}'
# → 1,106 ha plot, 98 ha loss after 31 Dec 2020 (2021–24), containing cell risk 0.99
```

## Data issues found and handled
- **Two sensor breaks** in the NDVI record: AVHRR → VIIRS S-NPP (2014) and S-NPP → NOAA-20 (2023, not in the
  original spec; detected from file names during validation, then caught again by the drift report).
- Cloud: 8 sampled days per month, maximum-value composite, cell needs ≥ 50 % valid pixels.
- Hansen loss is measured on 2000-forest pixels only, so loss fractions share one baseline.
- Label prevalence shifts from ~4 % to ~20 % after 2012 (real increase plus improved Hansen detection).

## Design decisions
- **Why NDVI, not Sentinel-1 radar.** Radar sees through cloud and is what EO job postings ask for, but the
  resilience question needs a ≥ 20-year monthly record (S1 starts 2014, and S1-B's failure left a 2021–24 gap
  over Africa), and basin-scale S1 at 10 m is terabytes with no unauthenticated source reachable from this build.
  S1 change detection is the natural next module for plot-level `/plot-check`, where it would replace annual
  Hansen with near-real-time alerts.
- **Why 0.25°.** Matches PERSIANN; gives 25 × 25 NDVI pixels per cell to beat cloud noise.
- **Why PR-AUC and precision@10 %.** Positives are rare in training, and field capacity is a fixed share of cells.

## Limitations
No published tipping threshold exists for the Congo and none is estimated here. NDVI saturates over dense
canopy (vegetation optical depth is the upgrade). Hansen "loss" ≠ EUDR "deforestation". No fire, roads or
concession data yet. Docker image defined but not built in the development sandbox (no Docker daemon); the
same pinned requirements were installed in a clean venv and the API was smoke-tested end to end.
