import tqdm
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from SiPMStudio.processing.functions import double_exp_release, exp_charge, gaussian


def time_constants(times, one_waves, lback=10, lfor=500, verbose=False):
    short_taus = []
    long_taus = []
    charge_taus = []
    for i, wave in tqdm.tqdm(enumerate(one_waves), total=len(one_waves), disable=~verbose):
        peak_info = find_peaks(wave, height=250, distance=75, width=4)
        if (len(peak_info[0]) > 1) | (len(peak_info[0]) == 0): continue
        charge_time, release_time = times[peak_info[0][0]-lback:peak_info[0][0]], times[peak_info[0][0]:lfor]
        charge_form, release_form = wave[peak_info[0][0]-lback:peak_info[0][0]], wave[peak_info[0][0]:lfor]
        if (len(charge_form) == 0) | (len(release_form) == 0): continue
        try:
            release_coeffs, release_cov = curve_fit(double_exp_release, release_time, release_form, p0=[50, 200, 5e6, 500, 7])
            charge_coeffs, charge_cov = curve_fit(exp_charge, charge_time, charge_form, p0=[600, 92, 5])
            short_taus.append(min(release_coeffs[-1], release_coeffs[-2]))
            long_taus.append(max(release_coeffs[-1], release_coeffs[-2]))
            charge_taus.append(charge_coeffs[-1])
        except RuntimeError:
            continue
    return short_taus, long_taus, charge_taus


def fit_exp(x, y):
    x_new = x[y > 0]
    y_new = y[y > 0]
    ln_y = np.log(y_new)
    slope, intercept, r, p, stderr = linregress(x_new, ln_y)
    scale = (1 / (1e-9*1e3))
    return abs(slope)*scale, stderr*scale, slope, intercept