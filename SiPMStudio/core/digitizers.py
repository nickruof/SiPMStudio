from .data_loading import DataLoader


class Digitizer(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def format_data(self, waves=False, rows=None):
        pass


class CAENDT5730(Digitizer):

    def __init__(self, *args, **kwargs):
        self.id = None
        self.model_name = "CAENDT5730"
        self.adc_bitcount = 14
        self.sample_rate = 500e6
        self.v_range = 0.0

        self.e_cal = None
        self.int_window = None
        self.parameters = ["TIMETAG", "E_SHORT", "E_LONG"]
        super().__init__(*args, **kwargs)

    def format_data(self, waves=False, rows=None):
        if self.df_data is None:
            return None
        if rows is None:
            rows = []
        if len(rows) == 2:
            if waves:
                params_frame = self.df_data.iloc[rows[0]:rows[1], :3]
                params_frame.columns = self.parameters
                waves_frame = self.df_data.iloc[rows[0]:rows[1], 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[rows[0]:rows[1], :3]
                params_frame.columns = self.parameters
                return params_frame
        elif len(rows) == 1:
            if waves:
                params_frame = self.df_data.iloc[rows[0]:, :3]
                params_frame.columns = self.parameters
                waves_frame = self.df_data.iloc[rows[0]:, 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[rows[0]:, :3]
                params_frame.columns = self.parameters
                return params_frame
        else:
            if waves:
                params_frame = self.df_data.iloc[:, :3]
                params_frame.columns = self.parameters
                waves_frame = self.df_data.iloc[:, 4:]
                return waves_frame
            else:
                params_frame = self.df_data.iloc[:, :3]
                params_frame.columns = self.parameters
                return params_frame

    def input_settings(self, settings):
        self.id = settings["id"]
        self.v_range = settings["v_range"]
        self.e_cal = settings["e_cal"]
        self.int_window = settings["int_window"]

    def parse_xml(self, xmlfile):
        pass

