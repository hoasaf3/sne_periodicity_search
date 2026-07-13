import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

output_dir = "lasair_lightcurves"
ztf_fp_output_dir = "ztf_fp_lightcurves"

ZTF_FP_FILTERS = {
    "ZTF_g": {"fid": 1, "label": "g-Detections", "color": "g"},
    "ZTF_r": {"fid": 2, "label": "r-Detections", "color": "r"},
    "ZTF_i": {"fid": 3, "label": "i-Detections", "color": "orange"},
}
MAG_ERROR_FACTOR = 2.5 / np.log(10)


def get_lightcurve_from_file(name):
    filepath = f"{output_dir}/{name}.json"
    if not os.path.exists(filepath):
        return []

    with open(filepath) as f:
        lc = json.load(f)  
    return lc['candidates']


def plot_lightcurve(name, obs_jd, min_detections=0, epoch_after_discovery=None):
    """
    Plot lightcurve for object with given name.
    min_detections: minimum number of detections to plot the lightcurve
    """
    points = get_lightcurve_from_file(name)
    jd, mag, magerr, detections, g_bands = [], [], [], [], []

    if epoch_after_discovery:
        max_jd = obs_jd + epoch_after_discovery
        points = [p for p in points if p['jd'] <= max_jd]

    for point in points:
        if 'candid' in point:  # detection
            jd.append(point['jd'])
            mag.append(point['magpsf'])
            magerr.append(point['sigmapsf'])
            detections.append(True)
            g_bands.append(point['fid'] == 1)  # list of flags. fid=1 green, fid=2 red
        else:  # non-detection
            jd.append(point['jd'])
            mag.append(point['diffmaglim'])
            magerr.append(None)
            detections.append(False)

    if sum(detections) < min_detections:
        print(f"Not enough detections for {name} ({sum(detections)}). Skipping plot.")
        return

    jd = pd.Series(jd-obs_jd)
    mag = pd.Series(mag)
    magerr = pd.Series(magerr)
    detections = pd.Series(detections)
    g_bands = pd.Series(g_bands)

    
    plt.figure(figsize=(10,6))
    plt.errorbar(jd[detections][g_bands], mag[detections][g_bands], yerr=magerr[detections][g_bands],
                 fmt='o', label='g-Detections', color='g')
    plt.errorbar(jd[detections][~g_bands], mag[detections][~g_bands], yerr=magerr[detections][~g_bands],
                 fmt='o', label='r-Detections', color='r')
    plt.scatter(jd[~detections], mag[~detections], marker='v', color='black', label='Non-detections')
    plt.gca().invert_yaxis()
    plt.xlabel('Days since discovery')
    plt.ylabel('Magnitude')
    plt.title(f'{name}')
    plt.legend()
    plt.show()


def get_fp_lightcurve_from_file(name, fp_output_dir=ztf_fp_output_dir):
    filepath = f"{fp_output_dir}/{name}.json"
    if not os.path.exists(filepath):
        return []

    with open(filepath) as f:
        lc = json.load(f)
    return lc.get("lightcurve", [])


def plot_lightcurve_fp(
    name,
    obs_jd=None,
    min_detections=0,
    epoch_after_discovery=None,
    detection_snr=5.0,
    fp_output_dir=ztf_fp_output_dir,
):
    """
    Plot ZTF forced-photometry detections for object with given name.

    Only positive forced-flux measurements with forcediffimsnr >= detection_snr
    are plotted. If obs_jd is provided, the x-axis is days since discovery;
    otherwise it is JD.
    """
    points = get_fp_lightcurve_from_file(name, fp_output_dir=fp_output_dir)
    rows = []

    if obs_jd is not None and epoch_after_discovery:
        max_jd = obs_jd + epoch_after_discovery
        points = [p for p in points if p.get("jd") is not None and p["jd"] <= max_jd]

    for point in points:
        filter_name = point.get("filter")
        band = ZTF_FP_FILTERS.get(filter_name)
        if band is None:
            continue

        flux = point.get("forcediffimflux")
        flux_unc = point.get("forcediffimfluxunc")
        zpdiff = point.get("zpdiff")
        jd = point.get("jd")
        snr = point.get("forcediffimsnr")

        if flux is None or flux_unc is None or zpdiff is None or jd is None:
            continue

        flux = float(flux)
        flux_unc = float(flux_unc)
        zpdiff = float(zpdiff)
        jd = float(jd)
        snr = float(snr) if snr is not None else flux / flux_unc if flux_unc > 0 else np.nan

        if flux <= 0 or flux_unc <= 0 or not np.isfinite(snr) or snr < detection_snr:
            continue

        mag = zpdiff - 2.5 * np.log10(flux)
        magerr = MAG_ERROR_FACTOR * flux_unc / flux
        if not np.isfinite(mag) or not np.isfinite(magerr):
            continue

        rows.append(
            {
                "jd": jd,
                "x": jd - obs_jd if obs_jd is not None else jd,
                "mag": mag,
                "magerr": magerr,
                "filter": filter_name,
                "fid": band["fid"],
                "snr": snr,
            }
        )

    if len(rows) < min_detections:
        print(f"Not enough detections for {name} ({len(rows)}). Skipping plot.")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No FP detections found for {name}.")
        return

    plt.figure(figsize=(10, 6))
    for filter_name, band in ZTF_FP_FILTERS.items():
        band_df = df[df["filter"] == filter_name]
        if band_df.empty:
            continue
        plt.errorbar(
            band_df["x"],
            band_df["mag"],
            yerr=band_df["magerr"],
            fmt="o",
            label=band["label"],
            color=band["color"],
        )

    plt.gca().invert_yaxis()
    plt.xlabel("Days since discovery" if obs_jd is not None else "JD")
    plt.ylabel("Forced-photometry magnitude")
    plt.title(f"{name} ZTF FP")
    plt.legend()
    plt.show()
