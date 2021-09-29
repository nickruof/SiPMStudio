import tqdm
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from SiPMStudio.processing.functions import double_exp_release, exp_charge, gaussian


def time_constants(times, one_waves, lback=10, lfor=500):
    short_taus = []
    long_taus = []
    charge_taus = []
    for i, wave in tqdm.tqdm(enumerate(one_waves), total=len(one_waves)):
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


def spectrum_peaks(n, bin_centers, guess, sigma):
    bin_width = bin_centers[1] - bin_centers[0]
    bin_range = sigma / bin_width / 2

    peak_locs = []
    for center in guess:
        min_dist = bin_centers - center
        idx = np.where(min_dist == min(min_dist))[0][0]
        coeffs, covs = curve_fit(
                                    gaussian,
                                    bin_centers[idx-bin_range:idx+bin_range],
                                    n[idx-bin_range:idx+bin_range],
                                    p0=[max(n[idx-bin_range:idx+bin_range]), center, sigma]
                                )
        peak_locs.append(coeffs[1])
    return peak_locs
        
