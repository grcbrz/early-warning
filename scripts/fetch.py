"""Download raw inputs: python scripts/fetch.py {ndvi|precip|hansen}."""
import logging
import sys

from congo_ews.acquire import hansen, ndvi, precip
from congo_ews.config import END_YEAR, START_YEAR

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
what = sys.argv[1]
if what == "ndvi":
    print(len(ndvi.fetch_range(START_YEAR, END_YEAR, workers=16)))
elif what == "precip":
    print(len(precip.fetch_range(START_YEAR, END_YEAR, workers=12)))
elif what == "hansen":
    print(hansen.fetch_all())
else:
    sys.exit(f"unknown source {what}")
