import glob
import h5py
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.interpolate import interp1d
import scipy.constants as const
import SiPMStudio.plots.plots_base as plt_base

from SiPMStudio.analysis.dark import current_waveforms, integrate_current


class Sipm:
    def __init__(self, model, settings, dark_trig_settings=None, light_trig_settings=None):
        self.settings = {}
        with open(settings, "r") as json_file:
            self.settings["dark"] = json.load(json_file)

        if dark_trig_settings:
            with open(dark_trig_settings, "r") as json_file:
                self.settings["dark_trig"] = json.load(json_file)

        if light_trig_settings:
            with open(light_trig_settings, "r") as json_file:
                self.settings["light_trig"] = json.load(json_file)

        self.dark_names = ["gain", "dark_rate", "cross_talk", "afterpulse"]
        self.trig_names = ["n_ped", "n_total", "int_width", "n_samples"]

        self.files_dict = {"dark": {"t1": None, "t2": None}, "dark_trig": {"t1": None, "t2": None},
                           "light_trig": {"t1": None, "t2": None}}
        self.bias_voltages = {"dark": [], "dark_trig": [], "light_trig": []}
        self.data_dict = {"dark": {}, "dark_trig": {}, "light_trig": {}}

        self.model_name = model
        self.breakdown = 0
        self.micro_cap = 0
        self.area = 0

        self.initialize_files()
        self.initialize_data()

    def initialize_files(self):
        file_types = ["dark", "dark_trig", "light_trig"]
        for file in file_types:
            self.settings[file]["t1"] = glob.glob(self.settings[file]["output_path_t1"]+"/*.h5")
            self.settings[file]["t2"] = glob.glob(self.settings[file]["output_path_t2"]+"/*.h5")
            self.settings[file]["t1"].sort()
            self.settings[file]["t2"].sort()

    def initialize_data(self):
        file_types = ["dark", "dark_trig", "light_trig"]
        for file in file_types:
            param_names = None
            if file == "dark":
                param_names = self.dark_names
                self._load_data(file, "t2", param_names)
            elif file.endswith("trig"):
                param_names = self.trig_names
                self._load_data(file, "t1", param_names)
            else:
                print("Found unknown file type name in initialize data!")
                break

    def _load_data(self, file_type, tier, param_names):
        for file in self.settings[file_type][tier]:
            h5_file = h5py.File(file, "r")
            bias = h5_file["bias"][()]
            data_element = {}
            for data in param_names:
                dataset = h5_file[data]
                data_element[data] = dataset[()]
            self.data_dict[file_type][bias] = data_element
            self.bias_voltages[file_type].append(bias)

    def __str__(self):
        return str(self.data_dict)

    def __getitem__(self, item):
        data_type, param = item
        voltages = np.array(list(self.data_dict[data_type].keys()))
        param_data = []
        for voltage in voltages:
            param_data.append(self.data_dict[data_type][voltage][param])
        return np.array(voltages-self.breakdown), np.array(param_data)

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
        self.data = pd.read_csv(file_path, delimiter=" ", header=None)
        self.data.columns = columns

    def get_response(self, wavelength):
        if self.responsivity.empty:
            print("Load Responsivity Data!")
        else:
            return np.interp(x=wavelength, xp=self.responsivity["wavelength"], fp=self.responsivity["responsivity"])

    def calibrate(self, show=False):
        bias = self.data["bias"].to_numpy()
        photo_current = (self.data["light"] - self.data["dark"]).to_numpy() * 1.0e-9
        fit_data = linregress(bias, photo_current)
        slope, intercept = fit_data[0], fit_data[1]

        if show:
            plt.figure()
            plt.plot(bias, photo_current, "o", alpha=0.5, label="data")
            plt.plot(bias, slope*bias+intercept, color="magenta", alpha=0.85, label="linear fit")
            plt.xlabel("LED Bias (V)")
            plt.ylabel("Diode Photocurrent (A)")
            plt.grid(alpha=0.25)

        def func(voltage):
            return slope*voltage + intercept

        self.cal_func = func


class LightSource:

    def __init__(self, name):
        self.model_name = name
        self.wavelengths = None
        self.intensity_profile = None
        self.profile = None
        self.profile_norm = 1
        self.voltage = None

    def load_profile(self, file_path, show=False):
        profile_data = pd.read_csv(file_path, delimiter=", ", header=None, engine="python")
        self.wavelengths = profile_data[0]*1.0e-9
        self.intensity_profile = profile_data[1]
        self.profile = interp1d(self.wavelengths, self.intensity_profile, kind="linear")
        self._calculate_norm()
        if show:
            plt.figure()
            plt.plot(self.wavelengths, self.norm_profile(self.wavelengths), alpha=0.85)
            plt.xlabel("Wavelengths (m)")
            plt.ylabel("Relative Intensity")
            plt.grid(alpha=0.25)

    def _calculate_norm(self, spacing=1e-9):
        num_points = (max(self.wavelengths) - min(self.wavelengths)) / spacing
        x = np.linspace(min(self.wavelengths), max(self.wavelengths), round(num_points))
        y = self.profile(x)
        self.profile_norm = np.trapz(y, x)

    def norm_profile(self, wavelength):
        return self.profile(wavelength) / self.profile_norm


class IntegratingSphere:

    def __init__(self, sipm, diode, light_source):
        self.sipm = sipm
        self.diode = diode
        self.light_source = light_source
        self.photon_rate = 0 #N_photons / ns / mm^2

    def sphere_photon_rate(self, spacing=1e-9):
        photocurrent = self.diode.cal_func(self.light_source.voltage) / self.diode.area / 1.0e9
        num_points = (max(self.light_source.wavelengths) - min(self.light_source.wavelengths)) / spacing
        lam_x = np.linspace(min(self.light_source.wavelengths), max(self.light_source.wavelengths), int(num_points))
        self.photon_rate = photocurrent * np.trapz(self._photon_rate_integrand(lam_x), lam_x)

    def _photon_rate_integrand(self, lam):
        term1 = lam / (const.h * const.c)
        term2 = self.light_source.norm_profile(lam) / self.diode.get_response(lam)
        return term1 * term2