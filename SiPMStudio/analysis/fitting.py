import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import find_peaks

from SiPMStudio.processing.functions import sipm_fall
from SiPMStudio.processing.functions import sipm_rise


def super_pulse(waves_data):
    return np.mean(waves_data, axis=0)


def fit_super_pulse(time_series, wave, display=False):
    peaks = find_peaks(wave, distance=1000, width=4)
    peak_location = peaks[0][0]
    rise_region = wave.iloc[0:peak_location]
    fall_region = wave.iloc[peak_location:]
    rise_guess = [1, 1, 1, 1]
    fall_guess = [1, 1, 1, 1]
    coeffs_1, cov_1 = curve_fit(sipm_rise, time_series[0:peak_location], rise_region)
    coeffs_2, cov_2 = curve_fit(sipm_fall, time_series[peak_location:], fall_region)
    if display:
        t_rise = np.linspace(0, time_series[peak_location], 1000)
        t_fall = np.linspace(time_series[peak_location], time_series[-1], 1000)
        sns.set_style("whitegrid")
        plt.figure()
        plt.plot(time_series, wave)
        plt.plot(t_rise, sipm_rise(t_rise, coeffs_1[0], coeffs_1[1], coeffs_1[2]), color="red")
        plt.plot(t_fall, sipm_fall(t_fall, coeffs_2[0], coeffs_2[1], coeffs_2[2]), color="red")
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.legend(["Super-Pulse", "fit"])
    return coeffs_1, coeffs_2


def fit_hist(func, hist, bins, var=None, guess=None,
             poisson_ll=False, integral=None, method=None, bounds=None):
    """
    do a binned fit to a histogram (nonlinear least squares).
    can either do a poisson log-likelihood fit (jason's fave) or
    use curve_fit w/ an arbitrary function.

    - hist, bins, var : as in return value of pygama.utils.get_hist()
    - guess : initial parameter guesses. Should be optional -- we can auto-guess
              for many common functions. But not yet implemented.
    - poissonLL : use Poisson stats instead of the Gaussian approximation in
                  each bin. Requires integer stats. You must use parameter
                  bounds to make sure that func does not go negative over the
                  x-range of the histogram.
    - method, bounds : options to pass to scipy.optimize.minimize
    """
    if guess is None:
        print("auto-guessing not yet implemented, you must supply a guess.")
        return

    if poisson_ll:
        if var is not None and not np.array_equal(var, hist):
            print("variances are not appropriate for a poisson-LL fit!")
            return

        if method is None:
            method = "L-BFGS-B"

        result = minimize(_neg_poisson_log_like, x0=guess,
                          args=(func, hist, bins, integral),
                          method=method, bounds=bounds)

        coeffs, cov_matrix = result.x, result.hess_inv.todense()

    else:
        if var is None:
            var = hist # assume Poisson stats if variances are not provided

        # skip "okay" bins with content 0 +/- 0 to avoid div-by-0 error in curve_fit
        # if bin content is non-zero but var = 0 let the user see the warning
        zeros = (hist == 0)
        zero_errors = (var == 0)
        mask = ~(zeros & zero_errors)
        sigma = np.sqrt(var)[mask]
        hist = hist[mask]
        xvals = pgh.get_bin_centers(bins)[mask]
        if bounds is None:
            bounds = (-np.inf, np.inf)

        coeffs, cov_matrix = curve_fit(func, xvals, hist,
                                      p0=guess, sigma=sigma, bounds=bounds)

    return coeffs, cov_matrix


def _get_bin_centers(bins):
    """
    Returns an array of bin centers from an input array of bin edges.
    Works for non-uniform binning.
    """
    return (bins[:-1] + bins[1:]) / 2.


def _get_bin_widths(bins):
    """
    Returns an array of bin widths from an input array of bin edges.
    Works for non-uniform binning.
    """
    return bins[1:] - bins[:-1]


def _get_bin_estimates(pars, func, hist, bins, integral=None, **kwargs):
    """
    Bin expected means are estimated by f(bin_center)*bin_width. Supply an
    integrating function to compute the integral over the bin instead.
    """
    if integral is None:
        return func(_get_bin_centers(bins), *pars, **kwargs) * _get_bin_widths(bins)
    else:
        return integral(bins[1:], *pars, **kwargs) - integral(bins[:-1], *pars, **kwargs)


def _neg_poisson_log_like(pars, func, hist, bins, integral=None, **kwargs):
    """
    Wrapper to give me poisson neg log likelihoods of a histogram
        ln[ f(x)^n / n! exp(-f(x) ] = const + n ln(f(x)) - f(x)
    """
    mu = _get_bin_estimates(pars, func, hist, bins, integral, **kwargs)
    # func and/or integral should never give a negative value: let negative
    # values cause errors that get passed to the user. However, mu=0 is okay,
    # but causes problems for np.log(). When mu is zero there had better not be
    # any counts in the bins. So use this to pull the fit like crazy.
    return np.sum(mu - hist*np.log(mu+1.e-99))




