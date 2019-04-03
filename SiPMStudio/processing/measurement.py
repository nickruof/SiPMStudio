import numpy as np
import pandas as pd
from abc import ABC
import sys

import SiPMStudio.core.devices as gadget
import SiPMStudio.analysis.dark as sith
import SiPMStudio.analysis.light as jedi

class Measurement_Array(ABC):

    def __init__(self, settings=None):
        self.measurement_list = []
        self.settings = {}

        if settings is not None:
            self.settings = settings
        for key in settings:
            self.add(key, settings[key])


    def process(self):
        for measurment in self.measurement_list:
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
        elif fun_name in dir(jedi)
            self.measurement_list.append(
                Measurement(getattr(jedi, fun_name), self.settings[fun_name]))
        else:
            print("ERROR! unknown function: ", fun_name)
            sys.exit()

class Measurment:

    def __init__(self, function, fun_args={}):
        self.function = function
        self.fun_args = fun_args

    def process_block(self):
        return self.function(**self.fun_args)

class UtilityBelt:

    def __init__(self, gadgets={}):
        self.gadgets = gadgets

    def add(self, gadget_name, gadget_object):
        self.gadgets[gadget_name] = gadget_object



