"""Builds Table 4: compares each binary-population-synthesis model's predicted periodic
fraction (Ercolino et al. 2026) against f_5, the smallest intrinsic periodic fraction
for which our survey would recover three or more dual-band detections at least 5% of
the time, given that model's own period distribution.

For each model and each candidate population size X, this draws X periods from the
model's period distribution, draws semi-amplitudes uniformly up to 0.2 mag, looks up
each system's recovery probability in the Pij completeness grid (injection_recovery_grid.py),
and simulates whether it would actually be detected (a Bernoulli draw per system, since
the real survey returns an integer count, not an expectation value). Repeating this many
times per X gives P(N_detected >= target | X); f_5 is the smallest X / 34 for which that
probability exceeds 5%, for target = 2 and target = 3 (the paper uses 3, our number of
genuine dual-band detections -- target=2 is included for comparison).

Output: table4_full_per_X.csv (raw P(N_det >= target | X) curves) and
table4_summary_both_targets.csv (one f_5 per model per target).
"""
import time

import numpy as np
import pandas as pd

PIJ_CSV_PATH = "Pij_recovery_map.csv"
N_SURVEY = 34  # size of the dual-band sample (paper Eq. 19)
AMPLITUDE_MAX = 0.2
N_POP_DRAWS = 1000
TARGETS = [2, 3]
POPULATION_MC_SEED = 54321

ERCOLINO_PERIOD_FILES = {
    "k1m1e1": "ercolino_dist/k1m1e1_P_pdf.data",
    "k2m1e1": "ercolino_dist/k2m1e1_P_pdf.data",
    "k3m1e1": "ercolino_dist/k3m1e1_P_pdf.data",
    "k1m3e3": "ercolino_dist/k1m3e3_P_pdf.data",
    "k3m5e1": "ercolino_dist/k3m5e1_P_pdf.data",
    "k1m5e1": "ercolino_dist/k1m5e1_P_pdf.data",
    "k1m1e3": "ercolino_dist/k1m1e3_P_pdf.data",
}
MODELS_PREDICTIONS = {"k1m1e1": 15.6, "k2m1e1": 8.4, "k3m1e1": 5.6, "k1m3e3": 21.4,
                       "k3m5e1": 8.9, "k1m5e1": 26.8, "k1m1e3": 3.5}


def load_ercolino_period_distribution(path):
    dist = pd.read_csv(path, sep=r"\s+", comment="#")
    dist = dist[["period_days", "pdf"]].dropna()
    dist = dist[(dist["period_days"] > 0) & (dist["pdf"] > 0)].copy()
    periods = dist["period_days"].to_numpy(dtype=float)
    weights = dist["pdf"].to_numpy(dtype=float)
    return periods, weights / weights.sum()


def main():
    t0 = time.time()
    Pij_table = pd.read_csv(PIJ_CSV_PATH)

    # Vectorized bin lookup: for arrays of periods/amplitudes, find the matching
    # Pij_full_sample value for each, all at once (searchsorted on sorted bin edges).
    p_los = np.sort(Pij_table["P_lo"].unique())
    p_his = np.array([Pij_table.loc[Pij_table["P_lo"] == p, "P_hi"].iloc[0] for p in p_los])
    a_los = np.sort(Pij_table["A_lo"].unique())
    a_his = np.array([Pij_table.loc[Pij_table["A_lo"] == a, "A_hi"].iloc[0] for a in a_los])

    grid = np.zeros((len(p_los), len(a_los)))
    for i, p_lo in enumerate(p_los):
        for j, a_lo in enumerate(a_los):
            row = Pij_table[(Pij_table["P_lo"] == p_lo) & (Pij_table["A_lo"] == a_lo)]
            grid[i, j] = row.iloc[0]["Pij_full_sample"] if len(row) else 0.0

    def lookup_Pij_vectorized(periods, amplitudes):
        p_idx = np.searchsorted(p_los, periods, side="right") - 1
        a_idx = np.searchsorted(a_los, amplitudes, side="right") - 1
        valid = (
            (p_idx >= 0) & (p_idx < len(p_los)) & (a_idx >= 0) & (a_idx < len(a_los)) &
            (periods < p_his[np.clip(p_idx, 0, len(p_los) - 1)]) &
            (amplitudes < a_his[np.clip(a_idx, 0, len(a_los) - 1)])
        )
        out = np.zeros_like(periods, dtype=float)
        out[valid] = grid[p_idx[valid], a_idx[valid]]
        return out

    dists = {}
    for label, path in ERCOLINO_PERIOD_FILES.items():
        periods, weights = load_ercolino_period_distribution(path)
        dists[label] = {"periods": periods, "weights": weights}

    X_VALUES = np.arange(2, N_SURVEY + 1)
    print(f"N_SURVEY={N_SURVEY}, X range [{X_VALUES.min()},{X_VALUES.max()}]", flush=True)

    all_rows = []
    for model_i, model in enumerate(ERCOLINO_PERIOD_FILES):
        rng = np.random.default_rng(POPULATION_MC_SEED + model_i)
        d = dists[model]
        for X in X_VALUES:
            n_det_ge = {k: 0 for k in TARGETS}
            for _ in range(N_POP_DRAWS):
                periods = rng.choice(d["periods"], size=X, replace=True, p=d["weights"])
                amplitudes = rng.uniform(0.0, AMPLITUDE_MAX, size=X)
                pij_values = np.clip(lookup_Pij_vectorized(periods, amplitudes), 0.0, 1.0)
                n_det = int(np.sum(rng.binomial(1, pij_values)))
                for k in TARGETS:
                    if n_det >= k:
                        n_det_ge[k] += 1
            row = {"model": model, "X": X}
            for k in TARGETS:
                row[f"P_Ndet_ge_{k}"] = n_det_ge[k] / N_POP_DRAWS
            all_rows.append(row)
        print(f"  {model} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    full_table = pd.DataFrame(all_rows)
    full_table.to_csv("diagnostics/table4_full_per_X.csv", index=False)

    print("\n=== f5 summary, by target detection count ===")
    summary_rows = []
    for k in TARGETS:
        for model in ERCOLINO_PERIOD_FILES:
            sub = full_table[full_table["model"] == model].sort_values("X")
            hit = sub[sub[f"P_Ndet_ge_{k}"] > 0.05]
            if len(hit):
                min_X = int(hit.iloc[0]["X"])
                f5 = 100 * min_X / N_SURVEY
            else:
                min_X, f5 = np.nan, np.nan
            summary_rows.append({"target": k, "model": model,
                                  "prediction_pct": MODELS_PREDICTIONS[model],
                                  "min_X": min_X, "f5_pct": f5})
            print(f"  target={k} {model}: prediction={MODELS_PREDICTIONS[model]}%, min_X={min_X}, f5={f5}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("diagnostics/table4_summary_both_targets.csv", index=False)
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
