from .data_loading import DataLoader
from construct import Struct, Array, this, Int16ub, Int32ub, Int64ub


class Digitizer(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_parser = None

    def format_data(self, waves=False, rows=None):
        pass

    def update(self, params, waves):
        pass


class CAENDT5730(Digitizer):

    def __init__(self, *args, **kwargs):
        self.id = None
        self.model_name = "CAENDT5730"
        self.adc_bitcount = 14
        self.sample_rate = 500e6
        self.v_range = 2.0

        self.e_cal = None
        self.int_window = None
        self.parameters = ["TIMETAG", "E_LONG", "E_SHORT"]

        self.metadata_parser = Struct(
            "board" /  Int16ub,
            "channel" / Int16ub,
            "timestamp" / Int64ub,
            "energy" / Int16ub,
            "energy_short" / Int16ub,
            "flags" / Int32ub,
            "num_samples" / Int32ub,
            "waveform" / Array(this.num_samples, Int16ub)
        )
        super().__init__(*args, **kwargs)

    def initialize_data(self):
        if self.df_data is not None:
            self.df_data = self.df_data.rename(index=str, columns={0: "TIMETAG", 1: "E_SHORT", 2: "E_LONG", 3: "FLAGS"})
        else:
            raise LookupError("No Data Loaded!")

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

    def update(self, params_data, waves_data):
        if set(params_data.columns).issubset(self.df_data.columns):
            self.df_data.update(params_data)
        else:
            print(self.df_data.columns, params_data.columns)
            raise LookupError("Update Error, params columns don't match!")
        if set(waves_data.columns).issubset(self.df_data.columns):
            self.df_data.update(waves_data)
        else:
            print(waves_data.columns)
            raise LookupError("Update Error, waves columns don't match!")

    def input_settings(self, settings):
        self.id = settings["id"]
        self.v_range = settings["v_range"]
        self.e_cal = settings["e_cal"]
        self.int_window = settings["int_window"]

    def parse_xml(self, xmlfile):
        pass

