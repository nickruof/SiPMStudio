from abc import ABC
import pandas as pd

import SiPMStudio.processing.calculators as pc
import SiPMStudio.processing.transforms as pt


class Processor(ABC):
    """
    An object that stores a series of ProcessorBase classes to be independently run
    on a series of calcs and waves.  The calcs and waves are references to a pandas
    dataframe that is stored in the DataLoading base class of a digitizer object
    """

    def __init__(self, settings=None):
        self.file = ""
        self.proc_list = []
        self.digitizer = None
        self.calcs = []
        self.waves = []
        self.settings = {}

        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

    def set_processor(self, digitizer=None, rows=None):
        for processor in self.proc_list:
            if "file_name" in processor.fun_args.keys():
                processor.fun_args["file_name"] = self.file
        if digitizer is not None:
            self.digitizer = digitizer
        self.calcs = self.digitizer.format_data(waves=False, rows=rows)
        self.waves = self.digitizer.format_data(waves=True, rows=rows)

    def process(self):
        for processor in self.proc_list:
            if isinstance(processor, Calculator):
                self.calcs = processor.process_block(self.waves, self.calcs)
            elif isinstance(processor, Transformer):
                self.waves = processor.process_block(self.waves)
            else:
                raise TypeError("Couldn't identify processor type!")

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
            raise TypeError("ERROR! unknown function: ", fun_name)


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

