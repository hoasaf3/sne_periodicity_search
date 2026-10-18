"""Plots Fig. 7, the full-sample injection-recovery map: for every period/semi-amplitude
bin, the fraction of the dual-band sample for which an injected signal of that period
and amplitude would be recovered by the pipeline (computed by injection_recovery_grid.py).
The three dual-band detections (Table 1) are overplotted for reference, each placed at
the smaller of its two measured per-band semi-amplitudes.

Output: Pij_recovery_map.pdf, saved to both Figures/ and Overleaf/.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from periodicity_tools import get_decline_detections, analyze_candidate

PIJ_CSV_PATH = "Pij_recovery_map.csv"
TABLE1_NAMES = ["2020sgf", "2022esa", "2022jli"]

Pij_table = pd.read_csv(PIJ_CSV_PATH)

Pij_full_map = Pij_table.pivot(index="A_lo", columns="P_lo", values="Pij_full_sample").sort_index().sort_index(axis=1)
period_pairs = Pij_table[["P_lo", "P_hi"]].drop_duplicates().sort_values("P_lo")
period_edges = np.r_[period_pairs["P_lo"].iloc[0], period_pairs["P_hi"].to_numpy()]
period_centers = 0.5 * (period_edges[:-1] + period_edges[1:])
period_labels = [f"{int(lo)}-{int(hi)}" for lo, hi in zip(period_edges[:-1], period_edges[1:])]
amplitude_pairs = Pij_table[["A_lo", "A_hi"]].drop_duplicates().sort_values("A_lo")
amplitude_edges_plot = np.r_[amplitude_pairs["A_lo"].iloc[0], amplitude_pairs["A_hi"].to_numpy()]

# Recompute each Table 1 detection's own period/amplitude directly, rather than relying
# on any previously saved pipeline state.
detections = {}
for name in TABLE1_NAMES:
    decline_points = get_decline_detections(name, epoch_after_peak=1000)
    band_results = []
    for fid in [1, 2]:
        band_data = [p for p in decline_points if p.get("fid") == fid and "candid" in p]
        t = np.array([p["jd"] for p in band_data])
        m = np.array([p["magpsf"] for p in band_data])
        e = np.array([p["sigmapsf"] for p in band_data])
        res = analyze_candidate(name, t - t[0], m, e, poly_deg=3)
        band_results.append(res)
    detections[name] = band_results
    print(f"{name}: g P={band_results[0]['best_period']:.2f} A={band_results[0]['best_amplitude']:.3f} | "
          f"r P={band_results[1]['best_period']:.2f} A={band_results[1]['best_amplitude']:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
mesh = ax.pcolormesh(period_edges, amplitude_edges_plot, Pij_full_map.values,
                      shading="flat", cmap="viridis", vmin=0, vmax=0.5)
ax.set_ylim(amplitude_edges_plot[0], amplitude_edges_plot[-1])

bin_counts = defaultdict(int)
for name, br in detections.items():
    P = min(r["best_period"] for r in br)
    A = min(r["best_amplitude"] for r in br)
    hit = Pij_table[(Pij_table["P_lo"] <= P) & (P < Pij_table["P_hi"]) &
                    (Pij_table["A_lo"] <= A) & (A < Pij_table["A_hi"])]
    if hit.empty:
        print(f"{name} outside grid: P={P:.1f}, A={A:.3f}")
        continue
    row = hit.iloc[0]
    x_width = row["P_hi"] - row["P_lo"]
    x_center = 0.5 * (row["P_lo"] + row["P_hi"])
    y_center = 0.5 * (row["A_lo"] + row["A_hi"])
    key = (row["P_lo"], row["A_lo"])
    k = bin_counts[key]
    bin_counts[key] += 1
    offsets = [(-0.18, 1.00), (0.18, 1.00), (0.00, 1.10), (0.00, 0.91)]
    dx_frac, y_factor = offsets[k % len(offsets)]
    x = x_center + dx_frac * x_width
    y = y_center * y_factor
    ax.scatter(x, y, marker="*", s=220, c="red", edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate(name, xy=(x, y), xytext=(0, -5), textcoords="offset points", color="white",
                fontsize=6, weight="bold", ha="center", va="top", zorder=6)

ax.set_xticks(period_centers)
ax.set_xticklabels(period_labels)
ax.set_yticks(amplitude_edges_plot)
ax.set_yticklabels([f"{v:.2f}" for v in amplitude_edges_plot])
ax.hlines(amplitude_edges_plot, period_edges[0], period_edges[-1], colors="white", linewidth=0.35, alpha=0.35)
ax.minorticks_off()
ax.set_xlabel("Period bin [days]")
ax.set_ylabel("Semi-amplitude [mag]")
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("$P_{ij}$")
plt.tight_layout()
for path in ["Figures/Pij_recovery_map.pdf", "Overleaf/Pij_recovery_map.pdf"]:
    plt.savefig(path, dpi=300, bbox_inches="tight")
print("\nSaved Pij_recovery_map.pdf")
