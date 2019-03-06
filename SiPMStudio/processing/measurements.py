import numpy as np
import pandas as pd
import os

from scipy.sparse import diags

from SiPMStudio.core import data_loading
from SiPMStudio.core import digitizers
from SiPMStudio.core import devices
from SiPMStudio.calculations.helpers import detect_peaks
from SiPMStudio import functions



def collect_files(path, data_dir="UNFILTERED"):
    dirs_array = []
    file_array = []
    for dirpath, dirnames, filenames in os.walk(path):
        if dirnames:
            dirs_array.append(dirnames)
    runs = []
    waves = []

    for name in dirs_array[0]:
        if name.startswith("runs_"):
            runs.append(name)
        elif name.startswith("waves_"):
            waves.append(name)

    for run in runs:
        data_path = path+run+"/"+data_dir
        files = os.listdir(data_path)
        file_targets = []
        os.chdir(data_path)
        for file in files:
            if os.path.getsize(data_path+"/"+file) > 50:
                runs.append(data_path+"/"+file)
            else:
                pass
    for wave in waves:
        data_path = path+wave+"/"+data_dir
        files = os.listdir(data_path)
        file_targets = []
        os.chdir(data_path)
        for file in files:
            if os.path.getsize(data_path+"/"+file) > 50:
                waves.append(data_path+"/"+file)
            else:
                pass

    return runs, waves

def time_interval(df_data):
    interval = df_data["timetag"].iloc[-1] - df_data["timetag"].iloc[0]
    interval = interval * 1.0e-12
    return interval

def gain(params, digitizer, sipm, convert=False, sum_len=1):
    diffs = []
    gain_average = 1
    for i in range(0, len(self.params)-3, 3):
        diffs.append(params[i+3] - params[i])
    gain_average = sum(diffs[0:sum_len]) / float(len(diffs[0:sum_len]))
    if convert:
        gain_average = gain_average * digitizer.e_cal/1.6e-19
    sipm.gain.append(gain_average)    

def dark_count_rate(df_data, params, sipm):
    index = int(params[0] - (sipm.gain[-1]/2))
    bins = list(range(int(max(df_data["E_short"]))))
    bin_vals, _bin_edges = np.histogram(df_data["E_short"], bins=bins)
    dark_rate = (bin_vals.sum()/self.time_interval())
    sipm.dark_rate.append(dark_rate)
    #if units == "khz":
    #    dark_rate.ito(ureg.kilohertz)
    #elif units == "mhz":
    #    dark_rate.ito(ureg.megahertz)
    return dark_rate

def cross_talk(df_data, params, sipm):
    index1 = int(params[0] - sipm.gain[-1]/2)
    index2 = int(params[3] - sipm.gain[-1]/2)
    bins = list(range(int(max(df_data["E_short"]))))
    bin_vals, _bin_edges = np.histogram(df_data["E_short"], bins=bins)
    total_counts1 = sum(bin_vals[index1:])
    total_counts2 = sum(bin_vals[index2:])
    prob = total_counts2 / total_counts1
    sipm.cross_talk.append(prob)
    return prob

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
    M_diag = diags([-1, 1], [0, 1], shape=(len(all_times), len(all_times)))
    all_dts = M_diag @ all_times
    all_dts = np.delete(all_dts, -1)
    all_heights = np.delete(all_heights, -1)
    return all_dts, all_heights




