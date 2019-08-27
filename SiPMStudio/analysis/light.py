import numpy as np

from SiPMStudio.core import devices
from SiPMStudio.analysis.dark import excess_charge_factor

from scipy.stats import linregress
from scipy.integrate import trapz


def time_interval(params_data):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


def average_currents(dataloader, device, files, bias=None):
    if isinstance(device, devices.Sipm):
        currents = [None]*len(files)
        for i, file_name in enumerate(files):
            dataloader.load_data(file_name)
            currents[device.bias.index(bias[i])] = dataloader.current.mean()
            dataloader.clear_data()
        return currents
    elif isinstance(device, devices.Photodiode):
        currents = [None]*len(files)
        for i, file_name in enumerate(files):
            dataloader.load_data(file_name)
            currents[i] = dataloader.current.mean()
            dataloader.clear_data()
        return currents


def average_leakage(dataloader, sipm, bias, files):
    total_currents = average_currents(dataloader=dataloader, device=sipm, bias=bias, files=files)
    N = [sipm.pulse_rate[sipm.bias.index(voltage)] for voltage in bias]
    G = [sipm.gain_magnitude[sipm.bias.index(voltage)] for voltage in bias]
    q = 1.60217662e-19

    sipm_currents = np.multiply(G, N)
    sipm_currents = np.multiply(sipm_currents, q)
    leakage_currents = np.subtract(total_currents, sipm_currents)
    return leakage_currents


def to_photons(dataloader, diode, led, dark_files, light_files):
    dark_current = average_currents(dataloader, diode, dark_files)
    light_current = average_currents(dataloader, diode, light_files)

    h = 6.626e-34
    c = 3.0e8
    eta = diode.get_response(wavelength=led.wavelength)
    scale_factor = led.wavelength / (h * c * eta) / diode.area
    diff = np.subtract(light_current, dark_current)
    return diff * scale_factor


def diode_calibration(dataloader, diode, led, dark_files, light_files):
    dark_currents = average_currents(dataloader, diode, bias=None, files=dark_files)
    light_currents = average_currents(dataloader, diode, bias=None, files=light_files)
    photo_current = 1e6 * np.subtract(light_currents, dark_currents)
    photon_rate_per_area = to_photons(dataloader, diode, led, dark_files, light_files)
    slope, intercept, _rvalue, _pvalue, _stderr = linregress(photo_current, photon_rate_per_area)
    diode.cal_slope = slope
    diode.cal_intercept = intercept
    return photo_current, photon_rate_per_area


def continuous_pde(dataloader, sipm, diode, led, bias, dark_files, light_files):
    dark_currents = average_currents(dataloader, diode, bias=None, files=dark_files)
    light_currents = average_currents(dataloader, diode, bias=None, files=light_files)
    photo_currents = np.subtract(light_currents, dark_currents)
    incident_photon_rate = diode.photon_rate(photo_currents, sipm.area)
    detected_photon_rate = sipm.lcr_fit - sipm.ncr_fit
    sipm.pde = np.divide(detected_photon_rate / incident_photon_rate)
