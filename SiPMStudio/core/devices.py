import numpy as np
import pandas as pd


class sipm:

    def __init__(self, name, area):
        self.brand = name
        self.area = area
        self.bias = []
        self.gain = []
        self.dark_rate = []
        self.cross_talk = []
        self.after_pulse = []
        self.pde = []

        self.current = []
        self.I_current = []
        self.V_voltage = []

class photodiode:

    def __init__(self, name, area):
        self.brand = name
        self.area = area
        self.bias = None
        self.current = []
        self.responsivity = 0.0
