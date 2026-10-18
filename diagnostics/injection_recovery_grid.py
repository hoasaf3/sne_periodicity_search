"""Builds the injection-recovery completeness grid (Fig. 7's Pij map and Table 4's
population comparison) by injecting synthetic periodic signals into every dual-band
light curve in the sample and checking how often the pipeline recovers them.

This is the same computation as the "Pij grid" cell in periodicity_search.ipynb,
parallelized across candidates/period/amplitude bins with multiprocessing. Use this
script instead of the notebook cell when (re-)running the full grid from scratch --
with 1000 injections per candidate per bin it takes well over an hour single-threaded,
versus under two hours total here (adjust max_workers in main() to your machine).

Output: Pij_recovery_map.csv, in the same format the notebook cell produces.
"""
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import chi2

from periodicity_tools import get_decline_detections, analyze_candidate, compute_delta_chi2_curve

TNS_FILENAME = "tables/tns_public_objects_until_20250916.csv"
EPOCH_AFTER_PEAK = 1000
MIN_DETECTIONS_PER_BAND = 20
SINGLE_FREQ_P = 1e-4
SIG_THRESHOLD = chi2.ppf(1 - SINGLE_FREQ_P, df=2)
N_INJECTIONS = 1000
RNG_SEED = 12345

PERIOD_BINS = [(5, 15), (15, 25), (25, 35), (35, 45), (45, 55), (55, 65)]
AMPLITUDE_EDGES = np.round(np.arange(0.0, 0.2 + 0.01, 0.02), 5)


def build_candidate_library():
    tns = pd.read_csv(TNS_FILENAME, header=0)
    tns = tns[np.logical_and(tns["redshift"] <= 0.05, tns["declination"] > -60)]
    sn_typeids = [4, 5, 6, 7, 9, 106]
    tns_ibc = tns[np.isin(tns["typeid"], sn_typeids)]
    tns_ibc = tns_ibc[~tns_ibc["name"].isin(["2018ebt"])]

    entries_by_name = defaultdict(dict)
    for name in tns_ibc["name"]:
        decline_points = get_decline_detections(name, epoch_after_peak=EPOCH_AFTER_PEAK)
        for fid in [1, 2]:
            band_data = [p for p in decline_points if p.get("fid") == fid and "candid" in p]
            if len(band_data) < MIN_DETECTIONS_PER_BAND:
                continue
            t = np.array([p["jd"] for p in band_data])
            m = np.array([p["magpsf"] for p in band_data])
            e = np.array([p["sigmapsf"] for p in band_data])
            entries_by_name[name][fid] = {"name": name, "fid": fid, "days": t - t[0], "mags": m, "errs": e}

    dual_band_names = sorted(name for name, bands in entries_by_name.items() if 1 in bands and 2 in bands)
    return entries_by_name, dual_band_names


def has_peak_near_period(entry, target_period, period_tol):
    freq, dchi2 = compute_delta_chi2_curve(entry["days"], entry["mags"], entry["errs"], poly_deg=3)
    periods = 1.0 / freq
    mask = np.abs(periods - target_period) < period_tol
    if not np.any(mask):
        return False
    local_dchi2 = dchi2[mask]
    if len(local_dchi2) < 3:
        return np.nanmax(local_dchi2) > SIG_THRESHOLD
    local_max = (local_dchi2[1:-1] > local_dchi2[:-2]) & (local_dchi2[1:-1] > local_dchi2[2:])
    if not np.any(local_max):
        return np.nanmax(local_dchi2) > SIG_THRESHOLD
    return np.any(local_dchi2[1:-1][local_max] > SIG_THRESHOLD)


def recovered_in_both_bands(sim_entries, injected_period):
    results = []
    for entry in sim_entries:
        res = analyze_candidate(entry["name"], entry["days"], entry["mags"], entry["errs"], poly_deg=3)
        res["fid"] = entry["fid"]
        results.append(res)
    r1, r2 = results
    if not (
        r1["p_value_single_freq"] < SINGLE_FREQ_P and r2["p_value_single_freq"] < SINGLE_FREQ_P
        and r1["best_amplitude"] >= r1["mda"] and r2["best_amplitude"] >= r2["mda"]
    ):
        return False
    period_tol = max(r1["best_period"] ** 2 / r1["time_baseline"], r2["best_period"] ** 2 / r2["time_baseline"])
    if not (abs(r1["best_period"] - injected_period) < period_tol or abs(r2["best_period"] - injected_period) < period_tol):
        return False
    return all(has_peak_near_period(entry, injected_period, period_tol) for entry in sim_entries)


def make_injected_light_curve(entry, period, amplitude, rng):
    days, mags = np.asarray(entry["days"]), np.asarray(entry["mags"])
    phase = rng.uniform(0.0, 2.0 * np.pi)
    sim_entry = dict(entry)
    sim_entry["mags"] = mags + amplitude * np.sin(2.0 * np.pi * days / period + phase)
    return sim_entry


def run_one_candidate_cell(args):
    """Worker process: N_INJECTIONS trials for one (candidate, period bin, amplitude bin)."""
    name, p_lo, p_hi, a_lo, a_hi, entries_by_name, seed = args
    rng = np.random.default_rng(seed)
    recovered = 0
    for _ in range(N_INJECTIONS):
        injected_period = rng.uniform(p_lo, p_hi)
        injected_amplitude = rng.uniform(a_lo, a_hi)
        sim_entries = [
            make_injected_light_curve(entries_by_name[name][fid], injected_period, injected_amplitude, rng)
            for fid in (1, 2)
        ]
        recovered += int(recovered_in_both_bands(sim_entries, injected_period))
    return name, p_lo, p_hi, a_lo, a_hi, recovered


def main():
    t0 = time.time()
    entries_by_name, dual_band_names = build_candidate_library()
    print(f"Dual-band sample: {len(dual_band_names)} candidates", flush=True)

    baselines = {
        name: min(entries_by_name[name][1]["days"].max(), entries_by_name[name][2]["days"].max())
        for name in dual_band_names
    }

    tasks = []
    seed_seq = np.random.SeedSequence(RNG_SEED)
    for p_lo, p_hi in PERIOD_BINS:
        for a_lo, a_hi in zip(AMPLITUDE_EDGES[:-1], AMPLITUDE_EDGES[1:]):
            eligible = [name for name in dual_band_names if baselines[name] / 2.5 >= p_hi]
            for name in eligible:
                child_seed = seed_seq.spawn(1)[0]
                tasks.append((name, p_lo, p_hi, float(a_lo), float(a_hi), entries_by_name, child_seed))

    print(f"Total (candidate x bin) tasks: {len(tasks)}", flush=True)

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(run_one_candidate_cell, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                print(f"  {completed}/{len(tasks)} tasks done ({elapsed:.0f}s elapsed)", flush=True)

    rows_by_bin = defaultdict(lambda: {"recovered": 0, "trials": 0, "n_eligible": 0})
    for name, p_lo, p_hi, a_lo, a_hi, recovered in results:
        key = (p_lo, p_hi, a_lo, a_hi)
        rows_by_bin[key]["recovered"] += recovered
        rows_by_bin[key]["trials"] += N_INJECTIONS
        rows_by_bin[key]["n_eligible"] += 1

    n_dual = len(dual_band_names)
    rows = []
    for (p_lo, p_hi, a_lo, a_hi), agg in rows_by_bin.items():
        Pij_full_sample = agg["recovered"] / (n_dual * N_INJECTIONS)
        Pij_conditional = agg["recovered"] / agg["trials"] if agg["trials"] > 0 else np.nan
        rows.append({
            "P_lo": p_lo, "P_hi": p_hi, "A_lo": a_lo, "A_hi": a_hi,
            "Candidates": agg["n_eligible"], "RecoveredTrials": agg["recovered"],
            "InjectionTrials": agg["trials"], "Pij": Pij_conditional,
            "Pij_full_sample": Pij_full_sample,
        })

    Pij_table = pd.DataFrame(rows).sort_values(["P_lo", "A_lo"]).reset_index(drop=True)
    Pij_table.to_csv("Pij_recovery_map.csv", index=False)
    print(f"\nSaved Pij_recovery_map.csv (total time {time.time()-t0:.0f}s)", flush=True)
    print(Pij_table)


if __name__ == "__main__":
    main()
