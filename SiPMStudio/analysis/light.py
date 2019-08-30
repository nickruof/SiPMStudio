import numpy as np

from SiPMStudio.core import devices

import scipy.constants as const
from scipy.stats import linregress


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
    else:
        raise AttributeError("Unrecognized device!")


def average_leakage(dataloader, sipm, bias, files):
    total_currents = average_currents(dataloader=dataloader, device=sipm, bias=bias, files=files)
    N = [sipm.pulse_rate[sipm.bias.index(voltage)] for voltage in bias]
    G = [sipm.gain_magnitude[sipm.bias.index(voltage)] for voltage in bias]

    sipm_currents = np.multiply(G, N)
    sipm_currents = np.multiply(sipm_currents, const.e)
    leakage_currents = np.subtract(total_currents, sipm_currents)
    return leakage_currents


def to_photons(dataloader, diode, led, dark_files, light_files):
    dark_current = average_currents(dataloader, diode, dark_files)
    light_current = average_currents(dataloader, diode, light_files)

    eta = diode.get_response(wavelength=led.wavelength)
    scale_factor = led.wavelength / (const.h * const.c * eta) / diode.area
    diff = np.subtract(light_current, dark_current)
    return diff * scale_factor


def diode_calibration(dataloader, diode, led, dark_files=None, light_files=None, dark_data=None, light_data=None):
    dark_currents = None
    light_currents = None
    if (dark_data is None) and (light_data is None):
        dark_currents = average_currents(dataloader, diode, bias=None, files=dark_files)
        light_currents = average_currents(dataloader, diode, bias=None, files=light_files)
    elif (dark_files is None) and (light_files is None):
        dark_currents = dark_data
        light_currents = light_data
    else:
        raise AttributeError("Please only files or data!")
    photo_current = 1e6 * np.subtract(light_currents, dark_currents)
    photon_rate_per_area = to_photons(dataloader, diode, led, dark_files, light_files)
    slope, intercept, _rvalue, _pvalue, _stderr = linregress(photo_current, photon_rate_per_area)
    diode.cal_slope = slope
    diode.cal_intercept = intercept
    return photo_current, photon_rate_per_area


def photocurrent_pde(sipm_darks, sipm_lights, diode_darks, diode_lights, sipm, diode):
    diode_photocurrent = diode_lights - diode_darks
    photon_rate = sipm.area * (diode.cal_slope*diode_photocurrent*1000 + diode.cal_intercept)
    sipm_photocurrent = sipm_lights - sipm_darks
    ecf = np.divide(sipm.cross_talk, 100)
    ecf = ecf + 1
    expected_current = np.multiply(sipm.gain_magnitude, const.e*ecf*photon_rate)
    pde = sipm_photocurrent / expected_current
    sipm.pde = pde
    return pde


def continuous_pde(dataloader, sipm, diode, led, bias, dark_files, light_files):
    dark_currents = average_currents(dataloader, diode, bias=None, files=dark_files)
    light_currents = average_currents(dataloader, diode, bias=None, files=light_files)
    photo_currents = np.subtract(light_currents, dark_currents)
    incident_photon_rate = diode.photon_rate(photo_currents, sipm.area)
    detected_photon_rate = sipm.lcr_fit - sipm.ncr_fit
    sipm.pde = np.divide(detected_photon_rate / incident_photon_rate)
