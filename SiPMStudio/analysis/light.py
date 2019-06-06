import numpy as np

from SiPMStudio.core import devices
from SiPMStudio.analysis.dark import excess_charge_factor

from scipy.stats import linregress


def time_interval(params_data):
    interval = params_data["TIMETAG"].iloc[-1] - params_data["TIMETAG"].iloc[0]
    interval = interval * 1.0e-12
    return interval


def average_currents(dataloader, device, bias, files, many=True):
    if isinstance(device, devices.sipm):
        currents = [None]*len(files)
        for i, file_name in enumerate(files):
            dataloader.load_data(file_name)
            currents[device.bias.index(bias[i])] = dataloader.current.mean()
            dataloader.clear_data()
        return currents
    elif isinstance(device, devices.photodiode):
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
    dark_currents = average_currents(dataloader=dataloader, device=diode, bias=None, files=dark_files)
    light_currents = average_currents(dataloader=dataloader, device=diode, bias=None, files=light_files)
    h = 6.626e-34
    c = 3.0e8
    eta = diode.get_response(wavelength=led.wavelength)
    scale_factor = led.wavelength / (h * c * eta)
    diff = np.subtract(light_currents, dark_currents)
    return diff * scale_factor/diode.area


def diode_calibration(dataloader, diode, led, dark_files, light_files):
    dark_currents = average_currents(dataloader, diode, bias=None, files=dark_files)
    light_currents = average_currents(dataloader, diode, bias=None, files=light_files)
    photo_current = np.subtract(light_currents, dark_currents)
    photon_rate = to_photons(dataloader, diode, led, dark_files, light_files)
    slope, intercept, _rvalue, _pvalue, _stderr = linregress(photo_current, photon_rate)
    diode.cal_slope = slope
    diode.cal_intercept = intercept
    return photo_current, photon_rate


def continuous_pde(dataloader, sipm, diode, led, bias, dark_files, light_files):
    dark_sipm_currents = average_currents(dataloader=dataloader, device=sipm, bias=bias, files=dark_files[1:])
    light_sipm_currents = average_currents(dataloader=dataloader, device=sipm, bias=bias, files=light_files[1:])
    dark_diode_current = average_currents(dataloader=dataloader, device=diode, bias=None, files=[dark_files[0]])
    light_diode_current = average_currents(dataloader=dataloader, device=diode, bias=None, files=[light_files[0]])
    photocurrent = light_diode_current[0] - dark_diode_current[0]
    incident_photon_rate = diode.cal_slope*photocurrent + diode.cal_intercept

    q = 1.60217662e-19
    pde = np.subtract(light_sipm_currents, dark_sipm_currents)
    pde = np.divide(pde, ecf)
    pde = np.divide(pde, incident_photon_rate*sipm.area)
    pde = np.divide(pde, sipm.gain_magnitude)
    pde = np.divide(pde, q)
    sipm.pde = pde
    return pde
