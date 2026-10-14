"""AC6: API contract, including the EUDR plot check against synthetic 30 m Hansen tiles."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

os.environ["CONGO_EWS_LAZY"] = "1"  # don't load real artifacts at import time
from congo_ews.config import ANALYSIS_GRID  # noqa: E402
from congo_ews.serve.api import Store, create_app  # noqa: E402

RES = 0.00025
# Synthetic tile fragment: lon 20.00..20.10, lat -0.10..0.00 (inside Hansen tile 00N_020E).
W, N, SIZE = 20.0, 0.0, 400


def _write(path: Path, data: np.ndarray) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=SIZE, height=SIZE, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(W, N, RES, RES)) as dst:
        dst.write(data, 1)


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    hdir = tmp_path_factory.mktemp("hansen")
    tc = np.full((SIZE, SIZE), 80, np.uint8)
    ly = np.zeros((SIZE, SIZE), np.uint8)
    ly[:, :200] = 22   # west half (lon 20.00..20.05) lost in 2022
    ly[:50, 200:] = 15  # a strip lost in 2015, before the EUDR cut-off
    _write(hdir / "treecover2000_00N_020E.tif", tc)
    _write(hdir / "lossyear_00N_020E.tif", ly)

    g = ANALYSIS_GRID
    cells = [int(g.cell_id(24, 48)), int(g.cell_id(24, 47)), int(g.cell_id(10, 10))]
    years = [2022, 2023, 2024]
    panel = pd.DataFrame([{"year": y, "cell_idx": c, "lat": 0.1, "lon": 20.1, "loss_t": 0.01,
                           "ar1": 0.1, "variance": 0.005, "ar1_tau": np.nan, "label": 1.0 if y < 2024 else np.nan}
                          for y in years for c in cells])
    preds = pd.DataFrame([{"year": 2024, "cell_idx": c, "risk": r, "split": "live"}
                          for c, r in zip(cells, [0.9, 0.2, 0.5])])
    store = Store(panel, preds.assign(risk_percentile=preds.risk.rank(pct=True) * 100),
                  {"selected": "M1_hgb_pressure"}, hdir)
    return TestClient(create_app(store))


def polygon(w: float, s: float, e: float, n: float) -> dict:
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["latest_year"] == 2024


def test_risk_ranked_desc(client):
    body = client.get("/risk", params={"top_k": 2}).json()
    assert [c["risk"] for c in body] == [0.9, 0.5]


def test_risk_unknown_year_404(client):
    assert client.get("/risk", params={"year": 1990}).status_code == 404


def test_cell_detail_and_unknown_cell(client):
    cid = int(ANALYSIS_GRID.cell_id(24, 48))
    body = client.get(f"/cells/{cid}").json()
    assert len(body["history"]) == 3 and body["history"][0]["ar1_tau"] is None
    assert client.get("/cells/999999").status_code == 404


def test_plot_check_measures_post_cutoff_loss(client):
    r = client.post("/plot-check", json={"plot_id": "farm-1", "geometry": polygon(20.0, -0.1, 20.1, 0.0)})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["flag"] == "loss_after_cutoff"
    assert b["loss_after_cutoff_ha"] == pytest.approx(b["forest_2000_ha"] / 2, rel=0.01)
    assert set(b["loss_after_cutoff_by_year_ha"]) == {"2022"}
    assert b["loss_2001_2020_ha"] == pytest.approx(b["forest_2000_ha"] / 16, rel=0.02)
    # ~0.1 x 0.1 deg at the equator is ~12 300 ha.
    assert b["area_ha"] == pytest.approx(12_300, rel=0.01)
    assert "Screening only" in b["disclaimer"]


def test_plot_check_clean_plot(client):
    b = client.post("/plot-check", json={"geometry": polygon(20.06, -0.09, 20.09, -0.02)}).json()
    assert b["flag"] == "no_loss_detected" and b["loss_after_cutoff_ha"] == 0


@pytest.mark.parametrize("geometry", [
    {"type": "Point", "coordinates": [20.0, 0.0]},
    {"type": "Polygon", "coordinates": [[[20.0, -0.1], [20.1, -0.1], [20.1, 0.0]]]},  # too few
    {"type": "Polygon", "coordinates": [[[20.0, -0.1], [20.1, -0.1], [20.1, 0.0], [20.0, 0.0]]]},  # open
    polygon(20.0, -0.5, 20.5, 0.0),  # too large
    polygon(20.0, 0.0, 20.0, 0.1),  # degenerate
])
def test_plot_check_rejects_invalid_geometry(client, geometry):
    assert client.post("/plot-check", json={"geometry": geometry}).status_code == 422


def test_plot_check_outside_region(client):
    r = client.post("/plot-check", json={"geometry": polygon(-60.0, -3.0, -59.9, -2.9)})  # Amazon
    assert r.status_code == 422 and "outside" in r.json()["detail"]
