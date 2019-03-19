import numpy as np
import pandas as pd

from abc import ABC

class DataLoader(ABC):

    def __init__(self, df_data=None):
        if df_data is not None:
            self.load_data(df_data)
        else:
            self.df_data = None

    def load_data(self, df_data, chunksize=3000):
        if isinstance(df_data, pd.core.frame.DataFrame):
            self.df_data = df_data
        elif isinstance(df_data, str):
            self.df_data = pd.read_csv(df_data, delimiter=";", header=None, chunksize=chunksize)
        else:
            raise TypeError("DataType not recognized!")

    def clear_data(self):
        self.df_data = None


class Keithley2450(DataLoader):

    def __init__(self, *args, **kwargs):
        self.model_name = "Keithley2450"
        if self.df_data is not None:
            self.current = self.df_data["Reading"]
            self.voltage = self.df_data["Value"]
        super().__init__(*args, **kwargs)

    def load_data(self, df_data):
        if isinstance(df_data, pd.core.frame.DataFrame):
            self.df_data = df_data
        elif isinstance(df_data, str):
            self.df_data = pd.read_csv(df_data, delimiter=",", skiprows=7)
        else:
            raise TypeError("DataType not recognized!")
        if self.df_data is not None:
            self.current = self.df_data["Reading"]
            self.voltage = self.df_data["Value"]
