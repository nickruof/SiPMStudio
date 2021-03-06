from abc import ABC
import pandas as pd

import SiPMStudio.processing.calculators as pc
import SiPMStudio.processing.transforms as pt


class Processor(ABC):

    # TODO: Look into using in place transformations for the ProcessorBase class

    def __init__(self, settings=None):
        self.proc_list = []
        self.calcs = []
        self.waves = []
        self.settings = {}

        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

    def set_processor(self, waves_data, rows=None):
        self.waves = waves_data

    def process(self):
        for processor in self.proc_list:
            if isinstance(processor, Calculator):
                self.calcs = processor.process_block(self.waves, self.calcs)
            elif isinstance(processor, Transformer):
                self.waves = processor.process_block(self.waves)
            else:
                raise TypeError("Couldn't identify processor type!")
        return self.waves

    def add(self, fun_name, settings):
        if settings is None:
            settings = {}
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings

        if fun_name in dir(pc):
            self.proc_list.append(
                Calculator(getattr(pc, fun_name), self.settings[fun_name]))
        elif fun_name in dir(pt):
            self.proc_list.append(
                Transformer(getattr(pt, fun_name), self.settings[fun_name]))
        else:
            raise LookupError("ERROR! unknown function: ", fun_name)

    def clear(self):
        self.proc_list.clear()
        self.settings.clear()


class ProcessorBase(ABC):
    def __init__(self, function, fun_args={}):
        self.function = function
        self.fun_args = fun_args

    def process_block(self, waves, calcs):
        return self.function(waves, calcs, **self.fun_args)


class Calculator(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)

    def process_block(self, waves, calcs):
        return self.function(waves_data=waves, params_data=calcs, **self.fun_args)


class Transformer(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)

    def process_block(self, waves, calcs=None):
        return self.function(waves_data=waves, **self.fun_args)

