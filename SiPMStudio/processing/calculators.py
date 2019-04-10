import numpy as np
from scipy.optimize import curve_fit


def fit_multi_gauss(calcs, min_dist, min_height, display=False):
    bins = np.linspace(start=0, stop=max(calcs),
                       num=int(max(calcs))
    bin_vals, _bin_edges = np.histogram(calcs, bins=bins)
    peaks = detect_peaks(bin_vals, mpd=min_dist, mph=min_height)
    amplitudes = bin_vals[peaks]
    sigmas = [17]*len(peaks) #method needed to avoid hard coded sigma guess
    guess = []
    for i, peak in enumerate(peaks):
        guess.append(peak)
        guess.append(amplitudes[i])
        guess.append(sigmas[i])
    popt, pcov = curve_fit(multi_gauss, xdata=bins[:-1], ydata=bin_vals, p0=guess)
    fit = multi_gauss(bins[:-1], *popt)
    if display:
        plt.figure()
        plt.plot(bins[:-1], fit, color="red")
    return popt