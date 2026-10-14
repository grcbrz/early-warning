"""One-off: add the `platform` field to NDVI composites cached before it was recorded."""
import numpy as np

from congo_ews.acquire.ndvi import keys_for_month
from congo_ews.config import PATHS, SOURCES

for f in sorted((PATHS.raw / "ndvi").glob("*.npz")):
    with np.load(f) as z:
        data = {k: z[k] for k in z.files}
    if "platform" in data:
        continue
    y, m = map(int, f.stem.split("-"))
    keys = keys_for_month(y, m, SOURCES.ndvi_sample_days)
    data["platform"] = np.array(",".join(sorted({k.platform for k in keys})))
    np.savez_compressed(f, **data)
print("ok")
