"""Live checks against public buckets. Run with `pytest -m network`."""

import pytest

from congo_ews.acquire.ndvi import keys_for_month
from congo_ews.config import SOURCES
from congo_ews.serve.plot_check import check_plot

pytestmark = pytest.mark.network


def test_ndvi_listing_has_daily_files():
    assert len(keys_for_month(2020, 6, SOURCES.ndvi_sample_days)) == len(SOURCES.ndvi_sample_days)


def test_plot_check_reads_remote_hansen_tiles():
    g = {"type": "Polygon", "coordinates": [[[27.36, 2.86], [27.39, 2.86], [27.39, 2.89],
                                             [27.36, 2.89], [27.36, 2.86]]]}
    r = check_plot(g, hansen_dir=None)
    assert r.area_ha > 1000 and r.forest_2000_ha > 0
