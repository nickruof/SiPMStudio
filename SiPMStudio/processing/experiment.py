import numpy as np
import pandas as pd
from abc import ABC
#from multiprocessing import cpu_count, parallel
import sys

import SiPMStudio.analysis.dark as yin
import SiPMStudio.analysis.light as yang

class Experiment(ABC):

    def __init__(self, settings=None):
        self.measurement_list = []
        self.settings = {}

        if settings is not None:
        self.settings = settings
        for key in settings:
            self.add(key, settings[key])


    def process(self, num_blocks):
        for measurment in self.measurement_list:
            if isinstance(measurement, Measurement):
                p_result = processor.process_block()
            else:
                print("Unknown Measurment type!")

    def add(self, fun_name, settings={}):
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings

        if fun_name in dir(yin):
            self.measurement_list.append(
                Measurement(getattr(yin, fun_name), self.settings[fun_name]))
        elif fun_name in dir(yang)
            self.measurement_list.append(
                Measurement(getattr(yang, fun_name), self.settings[fun_name]))
        else:
            print("ERROR! unknown function: ", fun_name)
            sys.exit()

class Measurment:

    def __init__(self, function, fun_args={}):
        self.function = function
        self.fun_args = fun_args

    def process_block(self):
        return self.function(**self.fun_args)
