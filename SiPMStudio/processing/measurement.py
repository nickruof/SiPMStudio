import numpy as np
import pandas as pd
from abc import ABC
import sys

import SiPMStudio.core.devices as gadget
import SiPMStudio.analysis.dark as sith
import SiPMStudio.analysis.light as jedi


class MeasurementArray(ABC):

    def __init__(self, settings=None):
        self.measurement_list = []
        self.settings = {}
        self.digitizer = None
        self.calcs = []
        self.waves = []

        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

    def set_array(self, digitizer):
        self.digitizer = digitizer
        self.calcs = digitizer.format_data(waves=False)
        self.waves = digitizer.format_data(waves=True)

    def process(self):
        for measurement in self.measurement_list:
            if isinstance(measurement, Measurement):
                p_result = measurement.process_block()
            else:
                print("Unknown Measurment type!")

    def add(self, fun_name, settings={}):
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings

        if fun_name in dir(sith):
            self.measurement_list.append(
                Measurement(getattr(sith, fun_name), self.settings[fun_name]))
        elif fun_name in dir(jedi):
            self.measurement_list.append(
                Measurement(getattr(jedi, fun_name), self.settings[fun_name]))
        else:
            print("ERROR! unknown measurement function: ", fun_name)
            sys.exit()


class Measurement:

    def __init__(self, function, fun_args={}):
        self.function = function
        self.fun_args = fun_args

    def process_block(self):
        return self.function(**self.fun_args)


class UtilityBelt:

    def __init__(self, gadgets={}, data={}):
        self.gadgets = gadgets
        self.data = data

    def add_gadget(self, gadget_name, gadget_object):
        self.gadgets[gadget_name] = gadget_object

    def add_data(self, data_name, data_object):
        self.data[data_name] = data_object


