import numpy as np
import pandas as pd
from abc import ABC
from multiprocessing import cpu_count, Parallel
import sys

import SiPMStudio.processing.calculators as pc
import SiPMStudio.processing.transforms as pt


class Processor(ABC):

    def __init__(self, settings=None):
        self.proc_list = []
        self.device_list = []
        self.digitizer = None
        self.calcs = []
        self.waves = []

        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

    def set_processor(self, digitizer):
        if digitizer:
            self.digitizer = digitizer
        self.calcs = digitizer.format_data(waves=False)
        self.waves = digitizer.format_data(waves=True)

    def process(self, df_data):
        self.set_processor(digitizer)

        for processor in self.proc_list:
            if isinstance(processor, Calculator):
                p_result = processor.process_block(self.waves, self.calcs)
            elif isinstance(processor, Transformer):
                p_result = processor.process_block(self.waves)
            else:
                pass
        return p_result

    #def process_parallel(self, df_data):
    #    self.set_processor(digitizer)
    #    cores = cpu_count()
    #    partitions = cores
    #    data_split = np.array_split(data, partitions)
    #    pool = Pool(cores)
    #    data = pd.concat(pool.map(func, datasplit))


    def add(self, fun_name, settings={}):
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
            print("ERROR! unknown function: ", fun_name)
            sys.exit()

class ProcessorBase(ABC):
    def __init__(self, function, fun_args={});
        self.function = function
        self.fun_args = fun_args

    def process_block(self, waves, calcs):
        return self.function(waves, calcs, **self.fun_args)

class Calculator(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)

    def process_block(self, calcs):
        return self.function(waves, calcs, **self.fun_args)

class Transformer(ProcessorBase):
    def __init__(self, function, fun_args={}):
        super().__init__(function, fun_args)

    def process_block(self, waves):
        return self.function(waves, **self.fun_args)

    
