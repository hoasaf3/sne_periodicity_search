import numpy as np
from numpy.linalg import lstsq
from scipy.stats import chi2

from lasair_tools import get_lightcurve_from_file

MAX_GAP_JD = 50  # max allowed gap between detections, after which everything else is dropped

MAX_FREQ = 1/3
NYQUST_FACTOR = 5


def get_min_frequency(timespan):
    """Minimum frequency to search, set to 2 full cycles over the timespan of the data"""
    return 5/2 / timespan


def get_decline_detections(name, epoch_after_peak=None):
    points = get_lightcurve_from_file(name)
    if not points:
        return []

    detections = [point for point in points if 'candid' in point]
    # detections.sort(key=lambda x: x['jd'])

    if not detections:
        return []

    # ad-hoc to skip 2022xxf first peak (it shows 2 peaks)
    if name == '2022xxf':
        detections = [point for point in detections if point['jd'] - detections[0]['jd'] >= 40]

    # ad-hoc to skip 2019vsi first peak (it shows 2 peaks)
    if name == '2019vsi':
        detections = [point for point in detections if point['jd'] - detections[0]['jd'] >= 40]


    # Find peak
    min_mag_idx = np.argmin([p['magpsf'] for p in detections])
    decline_detections = detections[min_mag_idx:]
    
    # Find gaps > 100 days and cut off the tail
    for i in range(1, len(decline_detections)):
        previous_jd = decline_detections[i-1]['jd']
        current_jd = decline_detections[i]['jd']
        
        if current_jd - previous_jd > MAX_GAP_JD:
            # Gap found
            decline_detections = decline_detections[:i]
            break
            
    if epoch_after_peak:
        max_jd = decline_detections[0]['jd'] + epoch_after_peak
        decline_detections = [p for p in decline_detections if p['jd'] <= max_jd]
    
    return decline_detections


def _prepare_baseline(days, mags, errors, poly_deg):
    """
    Fit a polynomial baseline and prepare all necessary vairables for the frequency search
    """
    days = np.asarray(days)
    mags = np.asarray(mags)
    errors = np.asarray(errors)

    t_mean = np.mean(days)
    t_centered = days - t_mean

    T = days.max() - days.min()
    t_scaled = 2 * t_centered / T
    
    # poly_deg+1 columns of t_scaled^k
    t_matrix = np.column_stack([t_scaled**k for k in range(poly_deg + 1)])

    # We minimize chi^2 = sum[(y - model)^2 / sigma^2]
    # This is implemented by multiplying both t and y by 1/sigma
    w = 1.0 / errors
    tw_matrix = t_matrix * w[:, None]
    yw = mags * w

    # Find poly_coeffs s.t. tw_poly * poly_coeffs = yw
    poly_coeffs, *_ = lstsq(tw_matrix, yw, rcond=None)

    model_poly_vals = t_matrix @ poly_coeffs
    resids = mags - model_poly_vals
    chi2_poly = np.sum((resids / errors) ** 2)

    # Normalize errors to avoid underestimating errors
    N = len(mags)
    k = poly_deg + 1
    dof = N - k
    chi2_red = chi2_poly / dof
    scale_factor = max(1.0, np.sqrt(chi2_red))
    errors_rescaled = errors * scale_factor
    w = 1.0 / errors_rescaled
    tw_matrix = t_matrix * w[:, None]
    yw = mags * w
    poly_coeffs, *_ = lstsq(tw_matrix, yw, rcond=None)

    model_poly_vals = t_matrix @ poly_coeffs
    resids = mags - model_poly_vals
    chi2_poly = np.sum((resids / errors_rescaled) ** 2)

    # Minimal Detectable Amplitude
    mda = np.std(resids) / np.sqrt(len(resids))

    return {
        "days": days,
        "mags": mags,
        "resids": resids,
        "errors": errors,
        "normalized_errors": errors_rescaled,
        "t_mean": t_mean,
        "t_centered": t_centered,
        "t_scaled": t_scaled,
        "T": T,
        "poly_coeffs": poly_coeffs,
        "chi2_poly": chi2_poly,
        "mda": mda,
        "w": w,
        "yw": yw
    }


def compute_delta_chi2_curve(days, mags, errors,
                             poly_deg,
                             max_freq=MAX_FREQ,
                             nyquist_factor=NYQUST_FACTOR):
    """
    Calcualte the improvement in chi2 between baseline polynomial model and poly+sinusoid model.
    delta chi2 is chi2_poly - chi2_full, so bigger chi2 means better fit for full model.
    Returns (freq, delta_chi2_vals)
    """

    base = _prepare_baseline(days, mags, errors, poly_deg)

    T = base["T"]
    t_centered = base["t_centered"]
    t_scaled = base["t_scaled"]
    chi2_poly = base["chi2_poly"]
    w = base["w"]
    yw = base["yw"]
    mags = base["mags"]
    errors = base["normalized_errors"]

    f_min = get_min_frequency(T)
    f_max = max_freq
    delta_f = 1 / (nyquist_factor * T)

    frequencies = np.arange(f_min, f_max, delta_f)
    delta_chi2_vals = []

    for f in frequencies:

        omega = 2 * np.pi * f

        t_mat_full = np.column_stack([
            *[t_scaled**k for k in range(poly_deg + 1)],
            np.sin(omega * t_centered),
            np.cos(omega * t_centered)
        ])

        tw_mat_full = t_mat_full * w[:, None]
        coeffs_full, *_ = lstsq(tw_mat_full, yw, rcond=None)

        model_full_vals = t_mat_full @ coeffs_full
        chi2_full = np.sum(((mags - model_full_vals) / errors) ** 2)

        delta_chi2_vals.append(chi2_poly - chi2_full)

    return frequencies, np.array(delta_chi2_vals)


def analyze_candidate(name, days, mags, errors,
                      poly_deg=3,
                      max_freq=MAX_FREQ,
                      nyquist_factor=NYQUST_FACTOR):

    base = _prepare_baseline(days, mags, errors, poly_deg)
    errors = base["normalized_errors"]
    frequencies, delta_chi2_vals = compute_delta_chi2_curve(
        days, mags, errors,
        poly_deg,
        max_freq,
        nyquist_factor
    )

    # Find best freq (max delta chi2 , biggest improvement)
    best_idx = np.argmax(delta_chi2_vals)
    best_frequency = frequencies[best_idx]
    best_delta_chi2 = delta_chi2_vals[best_idx]

    # confidence interval in frequency space
    _, f_lo, f_hi, _ = find_peak_ci_from_dchi2(
        frequencies,
        delta_chi2_vals,
        conf=0.68,
        df=2
    )
    p_lo = 1 / f_hi if np.isfinite(f_hi) else np.nan
    p_hi = 1 / f_lo if np.isfinite(f_lo) else np.nan

    p_value = 1 - chi2.cdf(best_delta_chi2, df=2)

    omega_best = 2 * np.pi * best_frequency

    t_mean = base["t_mean"]
    t_centered = base["t_centered"]
    t_scaled = base["t_scaled"]
    T = base["T"]
    w = base["w"]
    yw = base["yw"]
    mags = base["mags"]
    errors = base["errors"]
    poly_coeffs = base["poly_coeffs"]

    # Find the coeffs of the best fit
    # This is done again, even though it was already done for this specific frequency during the wide search,
    # because now we need to store the reuslting coeffs (A, B). Since lstsq is deterministic, it's ok.
    # The alternataive is to store A,B for every trial frequency.
    t_mat_best = np.column_stack([
        *[t_scaled**k for k in range(poly_deg + 1)],
        np.sin(omega_best * t_centered),
        np.cos(omega_best * t_centered)
    ])

    tw_mat_best = t_mat_best * w[:, None]
    coeffs_full_best_freq, *_ = lstsq(tw_mat_best, yw, rcond=None)

    A = coeffs_full_best_freq[-2]
    B = coeffs_full_best_freq[-1]
    amplitude = np.sqrt(A**2 + B**2)

    # Define callable models for easy plotting later
    def poly_model(t_eval):
        t_eval = np.asarray(t_eval)
        t_c = t_eval - t_mean
        t_s = 2 * t_c / T
        X_eval = np.column_stack([
            t_s**k for k in range(poly_deg + 1)
        ])
        return X_eval @ poly_coeffs

    def full_model(t_eval):
        t_eval = np.asarray(t_eval)
        t_c = t_eval - t_mean
        t_s = 2 * t_c / T
        X_eval = np.column_stack([
            *[t_s**k for k in range(poly_deg + 1)],
            np.sin(omega_best * t_c),
            np.cos(omega_best * t_c)
        ])
        return X_eval @ coeffs_full_best_freq

    result = {
        "name": name,
        "best_frequency": best_frequency,
        "best_period": 1 / best_frequency,
        "period_lo": p_lo,
        "period_hi": p_hi,
        "best_amplitude": amplitude,
        "mda": base["mda"],
        "delta_chi2": best_delta_chi2,
        "chi2_poly": base["chi2_poly"],
        "p_value_single_freq": p_value,
        "poly_degree": poly_deg,
        "time_baseline": T,
        "resids": base["resids"],
        "poly_model": poly_model,
        "full_model": full_model,
        "poly_coeffs": poly_coeffs,
        "full_params": coeffs_full_best_freq,
    }

    return result


def find_peak_ci_from_dchi2(freq, dchi2, conf=0.68, df=2):
    """
    Finds the confidence interval around the strongest peak
    in Delta chi2 as a function of frequency.

    Returns:
        f_peak, f_lo, f_hi, level
    """
    freq = np.asarray(freq)
    dchi2 = np.asarray(dchi2)

    order = np.argsort(freq)
    freq = freq[order]
    dchi2 = dchi2[order]

    i_peak = np.nanargmax(dchi2)

    f_peak = freq[i_peak]
    peak_val = dchi2[i_peak]

    delta = chi2.ppf(conf, df=df)
    level = peak_val - delta

    # left intersection
    f_lo = np.nan
    for i in range(i_peak - 1, -1, -1):
        if dchi2[i] <= level <= dchi2[i + 1]:

            f_lo = np.interp(
                level,
                [dchi2[i], dchi2[i + 1]],
                [freq[i], freq[i + 1]]
            )
            break

    # right intersection
    f_hi = np.nan
    for i in range(i_peak, len(freq) - 1):
        if dchi2[i] >= level >= dchi2[i + 1]:

            f_hi = np.interp(
                level,
                [dchi2[i], dchi2[i + 1]],
                [freq[i], freq[i + 1]]
            )
            break

    return f_peak, f_lo, f_hi, level
