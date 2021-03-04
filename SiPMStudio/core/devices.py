import glob
import h5py
import json

import numpy as np
import pandas as pd
import uncertainties

from scipy.stats import linregress
import scipy.constants as const
import SiPMStudio.plots.plots_base as plt_base

from SiPMStudio.analysis.dark import current_waveforms, integrate_current


class Sipm:
    def __init__(self, settings, data_names=None):
        self.settings = None
        with open(settings, "r") as json_file:
            self.settings = json.load(json_file)

        if data_names:
            self.data_names = data_names
        else:
            data_names = ["gain", "dark_rate", "cross_talk", "afterpulse"]

        self.t1_list = glob.glob(self.settings["output_path_t1"] + "/*.h5").sort()
        self.file_list = glob.glob(self.settings["output_path_t2"] + "/*.h5").sort()
        self.bias_voltages = []
        self.data_names = data_names
        self.data_dict = {}
        self.breakdown = 0
        self.micro_cap = 0

        self.initialize_data()

    def initialize_data(self):
        for file_name in self.file_list:
            h5_file = h5py.File(file_name, "r")
            bias = h5_file["bias"][()]
            self.bias_voltages.append(bias)
            data_element = {}
            for data in self.data_names:
                dataset = h5_file[data]
                data_element[data] = dataset[()]
            self.data_dict[str(bias)] = data_element

    def __getitem__(self, name):
        bias_voltages = list(self.data_dict.keys())
        return_data = []
        for bias in bias_voltages:
            return_data.append(self.data_dict[bias][name])
        over_volts = np.array(list(map(float, bias_voltages))) - self.breakdown
        return over_volts, np.array(return_data)

    def __str__(self):
        return str(self.data_dict)

    def ph_histogram(self, ax, bias_voltage, file_type="t1", bins=1000, density=False, log=True):
        waveforms = None
        index = self.bias_voltages.index(bias_voltage)
        if file_type == "t1":
            h5_file = h5py.File(self.t1_list[index], "r")
            waveforms = h5_file["/raw/waveforms"][:] - h5_file["/raw/baselines"][:]
        elif file_type == "t2":
            h5_file = h5py.File(self.file_list[index], "r")
            waveforms = h5_file["/processed/waveforms"][:]
        else:
            raise AttributeError("Unrecognized file type: " + str(file_type))
        current_forms = current_waveforms(waveforms)
        charges = integrate_current(current_forms, 35, 200) * 1e12
        plt_base.plot_hist(ax, charges, bins=bins, density=density)
        ax.set_xlabel("Charge (pC)")
        ax.set_ylabel("Counts")
        if log:
            ax.set_yscale("log")


def extrapolate_breakdown(sipm):
    bias, gains = sipm["gain"]
    fit_data = linregress(bias, gains)
    slope = fit_data[0]
    intercept = fit_data[1]
    sipm.breakdown = -intercept / slope
    sipm.micro_cap = slope * const.e / 1e-15


class Photodiode:

    def __init__(self, name, area):
        self.brand = name
        self.area = area
        self.bias = None
        self.current = []
        self.responsivity = pd.DataFrame()
        self.cal_slope = 1

    def load_response(self, file_path):
        response_data = pd.read_csv(file_path, delimiter=", ", header=None, engine="python")
        self.responsivity["wavelength"] = np.multiply(response_data[0], 1e-9)
        self.responsivity["responsivity"] = response_data[1]

    def get_response(self, wavelength):
        if self.responsivity.empty:
            print("Load Responsivity Data!")
        elif isinstance(wavelength, uncertainties.core.Variable):
            return np.interp(x=wavelength.nominal_value, xp=self.responsivity["wavelength"], fp=self.responsivity["responsivity"])
        else:
            return np.interp(x=wavelength, xp=self.responsivity["wavelength"], fp=self.responsivity["responsivity"])

    def calibrate(self, light_source):
        response = self.get_response(light_source.wavelength)
        energy_per_photon = const.h*const.c / light_source.wavelength
        self.cal_slope = 1 / (energy_per_photon * response * self.area)

    def photon_rate(self, current, active_area):
        return active_area * self.cal_slope*current*1.0e-9


class Led:
    def __init__(self, name, wavelength):
        self.name = name
        self.wavelength = wavelength
