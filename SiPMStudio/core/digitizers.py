import numpy as np
import pandas as pd

from .data_loading import DataLoader


class Digitizer(DataLoader):

    def __init__(self, *args, **kwargs):
        self.id = None
        self.model_name = "CAENDT5730"
        self.adc_bitcount = 14
        self.sample_rate = 500e6
        self.v_range = 0.0

        self.e_cal = None
        self.int_window = None
        self.waves = False
        super().__init__(*args, **kwargs):

    def apply_settings(self, settings):
        if settings["digitizer"] == self.decoder_name:
            sk = settings.keys()
            if "window" in sk:
                self.window = True
                self.win_type = settings["window"]
            if "n_samp" in sk:
                self.n_samp = settings["n_samp"]
            if "n_blsamp" in sk:
                self.n_blsamp = settings["n_blsamp"]

    def format_data(self, waves=False):



class CAENDT5730(Digitizer):

    def __init__(self, *args, **kwargs):
        self.parameters = ["timetag", "E_short", "E_long"]
        super().__init__(*args, **kwargs):


    def format_data(self, waves=False):
        if waves:
            params_frame = self.df_data.iloc[:, :3]
            params_frame.columns = self.parameters
            waveforms = csv.iloc[:, 4:].copy()
            waveforms = waveforms.transpose().set_index(keys=np.array(range(0, 2*self.waveforms.shape[1], 2)))
            return waveforms
        else:
            params = self.df_data.iloc[:, :3]
            self.params_frame = pd.DataFrame(params, columns=self.parameters)
            return params

    def parse_xml(self, xmlfile):

