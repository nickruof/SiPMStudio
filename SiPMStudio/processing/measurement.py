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

    def run(self):
        for measurement in self.measurement_list:
            if isinstance(measurement, Measurement):
                p_result = measurement.process_block()
            else:
                raise TypeError("Unknown Measurment type!")

    def add(self, fun_name, settings={}, post_settings={}):
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings

        if fun_name in dir(sith):
            self.measurement_list.append(
                Measurement(getattr(sith, fun_name), self.settings[fun_name], post_settings))
        elif fun_name in dir(jedi):
            self.measurement_list.append(
                Measurement(getattr(jedi, fun_name), self.settings[fun_name], post_settings))
        else:
            raise TypeError("ERROR! unknown measurement function: ", fun_name)


class Measurement:

    def __init__(self, function, fun_args=None, post_args=None):
        self.function = function
        self.fun_args = fun_args
        self.post_args = post_args
        if post_args is not None:
            self.post_name = post_args["name"]
            self.utility_belt = post_args["belt"]

    def process_block(self):
        result = self.function(**self.fun_args)
        if self.post_args is not None:
            add_to_belt(self.post_name, self.utility_belt, result, "data")
        return result

    def add_to_belt(self, name, utility_belt, data, data_type="data"):
        if type == "data":
            utility_belt.add_data(data_name=name, data_object=data)
        elif type == "gadget":
            utility_belt.add_gadget(gadget_name=name, gadget_object=data)
        else:
            raise TypeError(type+" not recognized!")


class UtilityBelt:

    def __init__(self, gadgets=None, data=None):
        if self.gadgets is None:
            self.gadgets = {}
        if self.data is None:
            self.data = {}
        self.gadgets = gadgets
        self.data = data

    def add_gadget(self, gadget_name, gadget_object):
        self.gadgets[gadget_name] = gadget_object

    def add_data(self, data_name, data_object):
        self.data[data_name] = data_object

    def remove_gadget(self, gadget_name):
        del self.gadgets[gadget_name]

    def remove_data(self, data_name):
        del self.data[data_name]


