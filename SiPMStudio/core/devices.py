import numpy as np
import pandas as pd

import scipy.constants as const
import uncertainties
from uncertainties import unumpy


class Sipm:

    def __init__(self, name, area):
        self.brand = name
        self.area = area
        self.bias = []
        self.noise_power = []
        self.signal_power = []
        self.gain = []
        self.gain_magnitude = []
        self.pulse_rate = []
        self.dcr_fit = []
        self.cross_talk = []
        self.after_pulse = []
        self.pde = []

        self.ecf = []
        self.photon_rate = []
        self.current = []
        self.I_current = []
        self.V_voltage = []

    def dump_data(self):
        data_values = pd.DataFrame()
        data_values["bias"] = unumpy.nominal_values(self.bias)
        data_values["gain"] = unumpy.nominal_values(self.gain_magnitude)
        data_values["dcr_fit"] = unumpy.nominal_values(self.dcr_fit)
        data_values["cross_talk"] = unumpy.nominal_values(self.cross_talk)
        data_values["ecf"] =unumpy.nominal_values(self.ecf)
        data_values["pde"] = unumpy.nominal_values(self.pde)

        data_errors = pd.DataFrame()
        data_errors["bias"] = unumpy.std_devs(self.bias)
        data_errors["gain"] = unumpy.std_devs(self.gain_magnitude)
        data_errors["dcr_fit"] = unumpy.std_devs(self.dcr_fit)
        data_errors["cross_talk"] = unumpy.std_devs(self.cross_talk)
        data_errors["ecf"] = unumpy.std_devs(self.ecf)
        data_errors["pde"] = unumpy.std_devs(self.pde)

        return data_values, data_errors


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
