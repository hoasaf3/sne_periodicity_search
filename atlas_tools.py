import json
from pathlib import Path

import numpy as np
import pandas as pd

ATLAS_LIGHTCURVE_DIR = Path("atlas_lightcurves")
ATLAS_FILTER_TO_FID = {"c": 4, "o": 5}  # ATLAS cyan / orange, distinct from ZTF fids 1/2/3


def get_atlas_lightcurve_from_file(name, snr_threshold=5.0, lightcurve_dir=ATLAS_LIGHTCURVE_DIR):
    """Load a cached ATLAS forced-photometry light curve (written by fetch_atlas_2020sgf.py's
    write_atlas_lightcurve_to_file) and return detections per band as
    {fid: (jd, mag, err)}, fid 4=ATLAS-c, 5=ATLAS-o. Only keeps points with
    |uJy|/duJy >= snr_threshold (real detections, not upper limits).
    """
    filepath = Path(lightcurve_dir) / f"{name}.json"
    if not filepath.exists():
        return {}

    records = json.loads(filepath.read_text())
    df = pd.DataFrame(records)
    if df.empty:
        return {}

    dujy = pd.to_numeric(df.get("duJy"), errors="coerce")
    ujy = pd.to_numeric(df.get("uJy"), errors="coerce")
    # NOTE: must require uJy > 0 (real, positive flux), not abs(uJy)/duJy >= threshold --
    # a significant *negative* flux (subtraction artifact / bad reference) also passes
    # the abs() version and produces a nonsensical negative "magnitude". This bug also
    # exists in fetch_at_las_t.ipynb's plot_atlas_fp().
    det_mask = (dujy > 0) & (ujy / dujy >= snr_threshold)

    mag = pd.to_numeric(df["m"], errors="coerce")
    mag_err = pd.to_numeric(df["dm"], errors="coerce")
    mjd = pd.to_numeric(df["MJD"], errors="coerce")
    filt = df["F"].astype(str).str.lower()

    bands = {}
    for f_char, fid in ATLAS_FILTER_TO_FID.items():
        sel = det_mask & (filt == f_char) & mag.notna() & mag_err.notna() & mjd.notna()
        if sel.sum() == 0:
            continue
        jd = mjd[sel].to_numpy() + 2400000.5
        order = np.argsort(jd)
        bands[fid] = (jd[order], mag[sel].to_numpy()[order], mag_err[sel].to_numpy()[order])

    return bands
