import numpy as np
import pandas as pd
import os

from scipy.sparse import diags

from SiPMStudio.core import data_loading
from SiPMStudio.core import digitizers
from SiPMStudio.core import devices
from SiPMStudio.calculations.helpers import detect_peaks
from SiPMStudio import functions


def time_interval(df_data):
    interval = df_data["timetag"].iloc[-1] - df_data["timetag"].iloc[0]
    interval = interval * 1.0e-12
    return interval

def delay_times(params_data, wave_data, min_height, min_dist):
    all_dts = []
    all_times = []

    for i, wave in enumerate(wave_data):
        peaks = detect_peaks(wave, mph=min_height, mpd=min_dist)
        times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
        all_times = np.append(all_times, [times])
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    return all_dts

def heights(params_data, wave_data, min_height, min_dist):
    all_heights = []

    for i, wave in enumerate(wave_data):
        peaks = detect_peaks(wave, mph=min_height, mpd=min_dist)
        heights = wave_data.iloc[peaks, i].values
        all_heights = np.append(all_heights, [heights])
    all_heights = np.delete(all_heights, -1)
    return all_heights

def delay_time_vs_height(params_data, wave_data, min_height, min_dist):
    all_dts = []
    all_heights = []
    all_times = []

    for i, wave in enumerate(wave_data):
        peaks = detect_peaks(wave, mph=min_height, mpd=min_dist)
        times = np.add(params_data.iloc[i, 0]*10**-3, 2*peaks)
        heights = wave_data.iloc[peaks, i].values
        all_times = np.append(all_times, [times])
        all_heights = np.append(all_heights, [heights])
    if len(all_dts) == 0 or len(all_heights) == 0:
        print("No peaks found!")
        return all_dts, all_heights
    else:
        M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
        all_dts = M_diag @ all_times
        all_dts = np.delete(all_dts, -1)
        all_heights = np.delete(all_heights, -1)
        return all_dts, all_heights

def average_currents(dataloader, sipm, bias, files):
    currents = [None]*len(bias)
    for i, file_name in enumerate(files):
        dataloader.load_data(file_name)
        currents[np.where(sipm.bias = bias[i])] = dataloader.current.mean()
        dataloader.clear_data()
    return currents

def average_leakage(dataloader, sipm, bias, files):
    total_currents = average_currents(dataloader=dataloader, sipm=sipm, bias=bias, files=files)
    N = [sipm.dark_rate[np.where(sipm.bias = voltage)] for voltage in bias]
    G = [sipm.gain[np.where(sipm.bias = voltage)] for voltage in bias]
    q = 1.60217662e-19

    sipm_currents = np.multiply(G, N)
    sipm_currents = np.multiply(sipm_currents, q)
    leakage_currents = np.subtract(total_currents, sipm_currents)
    return leakage_currents

def excess_charge_factor(sipm):
    return np.divide(sipm.dark_rate, sipm.dcr_fit)

def to_photons(dataloader, diode, led, dark_files, light_files):
    dark_currents = average_currents(dataloader, dark_files)
    light_currents = average_currents(dataloader, light_files)
    h = 6.626e-34
    c = 3.0e8
    eta = diode.get_response(led.wavelength)
    scale_factor = led.wavelength / (h * c * eta)
    diff = np.subtract(light_currents, dark_currents)
    return diff * scale_factor

def continous_pde(dataloader, sipm, diode, led, bias, dark_files, light_files):
    dark_sipm_currents = average_currents(dark_files)
    light_sipm_currents = average_currents(light_files)
    incident_photons = to_photons(dataloader, diode, led, dark_files[0], light_files[0])
    ecf = excess_charge_factor(sipm)
    q = 1.60217662e-19
    pde = np.subtract(light_sipm_currents, dark_sipm_currents)
    pde = np.divide(pde, ecf)
    pde = np.divide(pde, incident_photons)
    pde = np.divide(pde, sipm.gain)
    pde = np.divide(pde, q)
    sipm.pde = pde
    return pde

