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
        data["dark_rate"] = self.dark_rate
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
        self.responsivity = pd.read_csv(file_path, delimiter=", ")
        self.responsivity.rename(columns={0:"wavelength", 1:"responsivity"})

    def get_response(self, wavelength):
        if not responsivity:
            print("Load Responsivity Data!")
        else:
            return np.interp(x=wavelength, xp=self.responsivity["wavelength"], fp=self.responsivity["responsivity"])

class led:
    def __init__(self, name, wavelength):
        self.name = name
        self.wavelength = wavelength
