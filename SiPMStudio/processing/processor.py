import SiPMStudio.processing.calculators as pc
import SiPMStudio.processing.transforms as pt


class Processor(object):

    # TODO: Look into using in place transformations for the ProcessorBase class

    def __init__(self, settings=None):
        self.proc_list = []
        self.outputs = {}
        self.settings = {}

        if settings is not None:
            self.settings = settings
            for key in settings:
                self.add(key, settings[key])

    def process(self):
        for processor in self.proc_list:
            if isinstance(processor, Calculator):
                self.calcs = processor.process_block(self.waves, self.calcs)
                if processor.output_name:
                    output_name = self.calcs
            elif isinstance(processor, Transformer):
                self.waves = processor.process_block(self.waves)
                if processor.output_name:
                    output_name = self.waves
            else:
                raise TypeError("Couldn't identify processor type!")
        return self.outputs

    def add(self, fun_name, settings):
        if settings is None:
            settings = {}
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings
        if fun_name in dir(pc):
            self.proc_list.append(
                ProcessorBase(getattr(pc, fun_name), self.outputs, self.settings[fun_name]))
        elif fun_name in dir(pt):
            self.proc_list.append(
                ProcessorBase(getattr(pt, fun_name), self.outputs, self.settings[fun_name]))
        else:
            raise LookupError(f"ERROR! unknown function: {fun_name}")
    
    def init_outputs(self, outputs):
        self.outputs = outputs

    def reset_outputs(self):
        self.outputs.clear()

    def clear(self):
        self.proc_list.clear()
        self.settings.clear()


class ProcessorBase(object):
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.fun_args = args
        self.fun_kwargs = kwargs

    def process_block(self):
        self.function(self.outputs, *self.fun_args, **self.fun_kwargs)
