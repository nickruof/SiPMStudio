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

    def run(self, utility_belt):
        for measurement in self.measurement_list:
            if isinstance(measurement, Measurement):
                p_result = measurement.process_block(utility_belt)
            else:
                raise TypeError("Unknown Measurment type!")

    def add(self, fun_name, settings=None, post_settings=None, retrieve_settings=None):
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings

        if fun_name in dir(sith):
            self.measurement_list.append(
                Measurement(getattr(sith, fun_name), self.settings[fun_name], post_settings, retrieve_settings))
        elif fun_name in dir(jedi):
            self.measurement_list.append(
                Measurement(getattr(jedi, fun_name), self.settings[fun_name], post_settings, retrieve_settings))
        else:
            raise TypeError("ERROR! unknown measurement function: ", fun_name)


class Measurement:

    def __init__(self, function, fun_args=None, post_args=None, retrieve_args=None):
        self.function = function
        self.fun_args = fun_args
        self.post_name = None
        self.retrieve = None
        if post_args is not None:
            self.post_name = post_args["name"]
        if retrieve_args is not None:
            self.retrieve = retrieve_args

    def process_block(self, utility_belt=None):
        if self.retrieve is not None:
            self.fun_args[self.retrieve["variable"]] = utility_belt[self.retrieve["name"]]
        result = self.function(**self.fun_args)
        if self.post_name is not None:
            utility_belt.add_data(self.post_name, result)
        return result


class UtilityBelt:

    def __init__(self, data=None):
        if data is None:
            self.data = {}
        else:
            self.data = data

    def __getitem__(self, name):
        return self.data[name]

    def set_belt(self, names):
        for name in names:
            self.data[name] = None

    def add_data(self, data_name, data_object):
        self.data[data_name] = data_object

    def retrieve_data(self, data_name):
        return self.data[data_name]

    def remove_data(self, data_name=None):
        if data_name is None:
            self.data = {}
        else:
            del self.data[data_name]


