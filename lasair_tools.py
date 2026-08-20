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

