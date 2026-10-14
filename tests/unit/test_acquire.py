"""Pure parsing helpers of the acquisition layer (no network)."""

import numpy as np

from congo_ews.acquire.hansen import tile_name, tiles_for_bbox
from congo_ews.acquire.ndvi import max_value_composite, parse_key
from congo_ews.config import BBox
from congo_ews.process.build import sensor_era


def test_parse_key_extracts_date_sensor_platform():
    k = parse_key("data/2023/VIIRS-Land_v001_JP113C1_NOAA-20_20230105_c20230110123456.nc")
    assert (k.date, k.sensor, k.platform) == ("20230105", "VIIRS", "NOAA-20")
    k = parse_key("data/2010/AVHRR-Land_v005_AVH13C1_NOAA-19_20100101_c20170406091314.nc")
    assert (k.sensor, k.platform) == ("AVHRR", "NOAA-19")
    assert parse_key("data/2010/index.html") is None


def test_sensor_era_splits_viirs_platforms_but_not_avhrr():
    assert sensor_era("AVHRR", "NOAA-16") == sensor_era("AVHRR", "NOAA-19") == "AVHRR"
    assert sensor_era("VIIRS", "S-NPP") != sensor_era("VIIRS", "NOAA-20")


def test_max_value_composite_handles_all_nan_pixels():
    stack = np.array([[[0.2, np.nan]], [[0.5, np.nan]]])
    out = max_value_composite(stack)
    assert out[0, 0] == np.float32(0.5) and np.isnan(out[0, 1])


def test_hansen_tiles_cover_congo_bbox():
    tiles = tiles_for_bbox(BBox(-10, 6, 8, 32))
    assert tile_name(10, 0) in tiles and tile_name(0, 30) in tiles and len(tiles) == 8
    assert tile_name(-10, 20) == "10S_020E"
