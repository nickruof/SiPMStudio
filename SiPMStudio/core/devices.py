import glob
import h5py
import json

import numpy as np
import pandas as pd
import uncertainties

from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.constants as const
import SiPMStudio.plots.plots_base as plt_base

from SiPMStudio.analysis.dark import current_waveforms, integrate_current


class Sipm:
    def __init__(self, model, settings, dark_trig_settings=None, light_trig_settings=None, data_names=None):
        self.settings = None
        self.settings["dark_trig"] = dark_trig_settings
        self.settings["light_trig"] = light_trig_settings
        with open(settings, "r") as json_file:
            self.settings["dark"] = json.load(json_file)

        self.dark_names = ["gain", "dark_rate", "cross_talk", "afterpulse"]
        self.trig_names = ["n_ped", "n_total"]

        self.files_dict = {"dark": {"t1": None, "t2": None}, "dark_trig": {"t1": None, "t2": None},
                           "light_trig": {"t1": None, "t2": None}}
        self.bias_voltages = {"dark": [], "dark_trig": [], "light_trig": []}
        self.data_dict = {"dark": {}, "dark_trig": {}, "light_trig": {}}

        self.model_name = ""
        self.breakdown = 0
        self.micro_cap = 0
        self.area = 0

        self.initialize_files()
        self.initialize_data()

    def initialize_files(self):
        file_types = ["dark", "dark_trig", "light_trig"]
        for file in file_types:
            self.settings[file]["t1"] = glob.glob(self.settings[file]["output_path_t1"] + "/*.h5").sort()
            self.settings[file]["t2"] = glob.glob(self.settings[file]["output_path_t2"] + "/*.h5").sort()

    def initialize_data(self):
        file_types = ["dark", "dark_trig", "light_trig"]
        for file in file_types:
            param_names = None
            if file == "dark":
                param_names = self.dark_names
            elif file.endswith("trig"):
                param_names = self.trig_names
            else:
                print("Found unknown file type name in initialize data!")
                break
            self._load_data(file, "t1", param_names)
            self._load_data(file, "t2", param_names)

    @staticmethod
    def _load_data(self, file_type, tier, param_names):
        for file in self.settings[file_type][tier]:
            h5_file = h5py.File(file, "r")
            bias = h5_file["bias"][()]
            data_element = {}
            for data in param_names:
                dataset = h5_file[data]
                data_element[data] = dataset[()]
            self.data[file_type][bias] = data_element
            self.bias_voltages[file_type].append(bias)

    def __str__(self):
        return str(self.data_dict)

    def __getitem__(self, item):
        data_type, param = item
        voltages = list(self.data_dict[data_type].keys())
        param_data = []
        for voltage in voltages:
            param_data.append(self.data_dict[data_type][voltage][param])
        return np.array(voltages), np.array(param_data)

    def ph_histogram(self, ax, bias_voltage, tier="t1", bins=1000, density=False, log=True):
        waveforms = None
        index = self.bias_voltages["dark"].index(bias_voltage)
        h5_file = h5py.File(self.files_dict["dark"][tier][index], "r")
        if tier == "t1":
            waveforms = h5_file["/raw/waveforms"][:] - h5_file["/raw/baselines"][:]
        elif tier == "t2":
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


def extrapolate_breakdown(sipm, data="dark"):
    bias, gains = sipm[[data, "gain"]]
    fit_data = linregress(bias, gains)
    slope = fit_data[0]
    intercept = fit_data[1]
    sipm.breakdown = -intercept / slope
    sipm.micro_cap = slope * const.e / 1e-15


class Photodiode:

    def __init__(self, name, area):
        self.model_name = name
        self.area = area
        self.bias = None
        self.current = []
        self.responsivity = pd.DataFrame()
        self.data = None
        self.cal_func = None

    def load_response(self, file_path):
        response_data = pd.read_csv(file_path, delimiter=", ", header=None, engine="python")
        self.responsivity["wavelength"] = np.multiply(response_data[0], 1e-9)
        self.responsivity["responsivity"] = response_data[1]

    def load_data(self, file_path):
        columns = ["bias", "dark", "dark_err", "light", "light_err"]
        self.data = pd.read_csv(file_path, delimiter=" ", header=None, columns=columns)

    def get_response(self, wavelength):
        if self.responsivity.empty:
            print("Load Responsivity Data!")
        elif isinstance(wavelength, uncertainties.core.Variable):
            return np.interp(x=wavelength.nominal_value, xp=self.responsivity["wavelength"], fp=self.responsivity["responsivity"])
        else:
            return np.interp(x=wavelength, xp=self.responsivity["wavelength"], fp=self.responsivity["responsivity"])

    def calibrate(self, light_source):
        bias = self.data["bias"].to_numpy()
        photo_current = (self.data["light"] - self.data["dark"]).to_numpy()
        fit_data = linregress(bias, photo_current)
        slope, intercept = fit_data[0], fit_data[1]

        def func(voltage):
            return slope*voltage + intercept

        self.cal_func = func


class LightSource:

    def __init__(self, name, wavelengths, intensity_profile):
        self.model_name = name
        self.wavelengths = wavelengths
        self.intensity_profile = intensity_profile
        self.profile = interp1d(wavelengths, intensity_profile, kind="linear")
        self.profile_norm = 1

    def calculate_norm(self, spacing=0.1):
        num_points = (max(self.wavelengths) - min(self.wavelengths) ) / spacing
        x = np.linspace(min(self.wavelengths), max(self.wavelengths), round(num_points))
        y = self.profile(x)
        self.profile_norm = np.cumsum(y) * spacing

    def norm_profile(self, wavelength):
        return self.profile(wavelength) / self.profile_norm