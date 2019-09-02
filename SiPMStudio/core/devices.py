import numpy as np
import pandas as pd

import scipy.constants as const
import uncertainties


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

        self.lcr_fit = []
        self.current = []
        self.I_current = []
        self.V_voltage = []

    def dump_data(self):
        data = pd.DataFrame()
        data["bias"] = self.bias
        data["gain"] = self.gain
        data["pulse_rate"] = self.pulse_rate
        data["dcr_fit"] = self.dcr_fit
        data["cross_talk"] = self.cross_talk
        data["after_pulse"] = self.after_pulse
        data["pde"] = self.pde
        return data


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
        self.cal_slope = light_source.wavelength / (self.area * const.h * const.c * response)

    def photon_rate(self, current, active_area):
        return active_area * self.cal_slope*current


class Led:
    def __init__(self, name, wavelength):
        self.name = name
        self.wavelength = wavelength
