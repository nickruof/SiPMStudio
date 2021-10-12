import json

import SiPMStudio.processing.calculators as pc
import SiPMStudio.processing.transforms as pt


class Processor(object):

    def __init__(self, settings=None):
        self.proc_list = []
        self.outputs = {}
        self.save_to_file = []
        self.settings = {}

        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

    def process(self):
        for processor in self.proc_list:
            if isinstance(processor, ProcessorBase):
                processor.process_block(self.outputs)
            else:
                raise TypeError("Couldn't identify processor type!")
        return {key: self.outputs[key] for key in self.save_to_file}

    def add(self, fun_name, settings):
        if settings is None:
            settings = {}
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings
        if fun_name in dir(pc):
            self.proc_list.append(
                ProcessorBase(getattr(pc, fun_name), **self.settings[fun_name]))
        elif fun_name in dir(pt):
            self.proc_list.append(
                ProcessorBase(getattr(pt, fun_name), **self.settings[fun_name]))
        else:
            raise LookupError(f"Unknown function: {fun_name}")
    
    def init_outputs(self, outputs):
        self.outputs = outputs

    def reset_outputs(self):
        self.outputs.clear()

    def add_to_file(self, var_name):
        if isinstance(var_name, str):
            self.save_to_file.append(var_name)
        elif isinstance(var_name, list):
            self.save_to_file += var_name
        else:
            raise TypeError(f"var_name of type {type(var_name)} must be str or list of strings")

    def clear(self):
        self.proc_list.clear()
        self.settings.clear()


class ProcessorBase(object):
    def __init__(self, function, **kwargs):
        self.function = function
        self.fun_kwargs = kwargs

    def process_block(self, outputs):
        self.function(outputs, **self.fun_kwargs)


def load_functions(proc_settings, processor):
    for key, params in proc_settings["processes"].items():
        processor.add(key, settings=params)
    for output in proc_settings["save_output"]:
        processor.save_to_file.append(output)
