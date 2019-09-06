import numpy as np

from SiPMStudio.core import devices
from SiPMStudio.processing.functions import line

import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import unumpy, ufloat


def time_interval(params_data):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


def average_currents(dataloader, device, files, bias=None):
    if isinstance(device, devices.Sipm):
        currents = [None]*len(files)
        errors = [None]*len(files)
        for i, file_name in enumerate(files):
            dataloader.load_data(file_name)
            currents[device.bias.index(bias[i])] = dataloader.current.mean()
            errors[device.bias.index(bias[i])] = dataloader.current.std()
            dataloader.clear_data()
        return unumpy.uarray(currents, errors)
    elif isinstance(device, devices.Photodiode):
        currents = [None]*len(files)
        errors = [None]*len(files)
        for i, file_name in enumerate(files):
            dataloader.load_data(file_name)
            currents[i] = dataloader.current.mean()
            errors[i] = dataloader.current.std()
            dataloader.clear_data()
            return unumpy.uarray(currents, errors)
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


def photocurrent_pde(sipm_darks, sipm_lights, diode_darks, diode_lights, sipm, diode):
    diode_photocurrent = diode_lights - diode_darks
    incident_photon_rate = diode.photon_rate(diode_photocurrent, sipm.area)
    sipm_photocurrent = sipm_lights - sipm_darks
    ecf = np.array(sipm.cross_talk) + 1
    expected_current = np.multiply(sipm.gain_magnitude, const.e*ecf*incident_photon_rate)
    pde = (sipm_photocurrent*1e-6) / expected_current
    sipm.pde = pde
    return pde


def continuous_pde(dataloader, sipm, diode, led, bias, dark_files, light_files):
    dark_currents = average_currents(dataloader, diode, bias=None, files=dark_files)
    light_currents = average_currents(dataloader, diode, bias=None, files=light_files)
    photo_currents = np.subtract(light_currents, dark_currents)
    incident_photon_rate = diode.photon_rate(photo_currents, sipm.area)
    detected_photon_rate = sipm.lcr_fit - sipm.ncr_fit
    sipm.pde = np.divide(detected_photon_rate / incident_photon_rate)
