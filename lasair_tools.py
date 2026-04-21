import os
import json
import matplotlib.pyplot as plt
import pandas as pd

output_dir = "lasair_lightcurves"


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