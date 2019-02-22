import numpy as np
import pandas as pd

from abc import ABC

class DataLoader(ABC):

	def __init__(self, df_data=None):

		if df_data is not None:
			self.load_data(df_data)
		else:
			self.df_data = None

	def load_data(self, df_data):

		if isinstance(df_data, pd.core.frame.DataFrame):
			self.df_data = df_data
		elif isinstance(df_data, str):
			self.df_data = pd.read_csv(df_data, delimiter=";", header=None, skiprows=1)
		else:
			raise TypeError("DataType not recognized!")

	def format_data(self, columns):
		self.df_data.columns = columns

	def clear_data(self):
		self.df_data = None
