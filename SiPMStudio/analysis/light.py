import numpy as np
import scipy.constants as const

from SiPMStudio.analysis.dark import list_files
from uncertainties import ufloat


def time_interval(params_data):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


# def average_currents(dataloader, device, files, bias=None):
#    if isinstance(device, devices.Sipm):
#        currents = [None]*len(files)
#        errors = [None]*len(files)
#        for i, file_name in enumerate(files):
#            dataloader.load_data(file_name)
#            currents[device.bias.index(bias[i])] = dataloader.current.mean()
#            errors[device.bias.index(bias[i])] = dataloader.current.std()
#            dataloader.clear_data()
#        return unumpy.uarray(currents, errors)
#    elif isinstance(device, devices.Photodiode):
#        currents = [None]*len(files)
#        errors = [None]*len(files)
#        for i, file_name in enumerate(files):
#            dataloader.load_data(file_name)
#            currents[i] = dataloader.current.mean()
#            errors[i] = dataloader.current.std()
#            dataloader.clear_data()
#            return unumpy.uarray(currents, errors)
#    else:
#        raise AttributeError("Unrecognized device!")

def load_current_measurements(path, scope):
    dark_files = list_files(path, "dark", ".csv")
    light_files = list_files(path, "light", ".csv")

    dark_currents = []
    light_currents = []

    for i, file in enumerate(dark_files):
        scope.load_data(path+"/"+file)
        current = ufloat(np.mean(scope.current), np.std(scope.current))
        dark_currents.append(current)
        scope.load_data(path+"/"+light_files[i])
        current = ufloat(np.mean(scope.current), np.std(scope.current))
        light_currents.append(current)

    return np.array(dark_currents)*1.0e6, np.array(light_currents)*1.0e6


def leakage_current(dark_currents, sipm):
    expected_current = np.array(sipm.gain_magnitude) * np.array(sipm.dcr_fit) * np.array(sipm.ecf) * 1.0e-6 * const.e
    leakage = np.array(dark_currents)*1.0e-6 - expected_current
    return leakage


def current_pde(sipm, dark_currents, light_currents, photon_rate):
    photocurrent = (light_currents - dark_currents) * 1.0e-6
    expected_current = const.e * (np.array(sipm.gain_magnitude) * photon_rate) * np.array(sipm.ecf)
    pde = photocurrent / expected_current
    sipm.pde = pde


# def photocurrent_pde(sipm_darks, sipm_lights, diode_darks, diode_lights, sipm, diode):
#    diode_photocurrent = diode_lights - diode_darks
#    incident_photon_rate = diode.photon_rate(diode_photocurrent, sipm.area)
#    sipm_photocurrent = sipm_lights - sipm_darks
#    ecf = np.array(sipm.cross_talk) + 1
#    expected_current = np.multiply(sipm.gain_magnitude, const.e*ecf*incident_photon_rate)
#    pde = (sipm_photocurrent*1e-6) / expected_current
#    sipm.pde = pde
#    return pde


# def continuous_pde(dataloader, sipm, diode, led, bias, dark_files, light_files):
#    dark_currents = average_currents(dataloader, diode, bias=None, files=dark_files)
#    light_currents = average_currents(dataloader, diode, bias=None, files=light_files)
#    photo_currents = np.subtract(light_currents, dark_currents)
#    incident_photon_rate = diode.photon_rate(photo_currents, sipm.area)
#    detected_photon_rate = sipm.lcr_fit - sipm.ncr_fit
#    sipm.pde = np.divide(detected_photon_rate / incident_photon_rate)
