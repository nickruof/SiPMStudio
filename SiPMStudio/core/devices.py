import numpy as np
import pandas as pd


class sipm:

    def __init__(self, name, area):
        self.brand = name
        self.area = area
        self.bias = []
        self.gain = []
        self.gain_magnitude = []
        self.pulse_rate = []
        self.dcr_fit = []
        self.cross_talk = []
        self.after_pulse = []
        self.pde = []

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

class photodiode:

    def __init__(self, name, area):
        self.brand = name
        self.area = area
        self.bias = None
        self.current = []
        self.responsivity = pd.DataFrame()

    def load_response(self, file_path):
        response_data = pd.read_csv(file_path, delimiter=", ", header=None, engine="python")
        self.responsivity["wavelength"] = np.multiply(response_data[0], 1e-9)
        self.responsivity["responsivity"] = response_data[1]


    def get_response(self, wavelength):
        if self.responsivity.empty:
            print("Load Responsivity Data!")
        else:
            return np.interp(x=wavelength, xp=self.responsivity["wavelength"], fp=self.responsivity["responsivity"])

class led:
    def __init__(self, name, wavelength):
        self.name = name
        self.wavelength = wavelength
