import numpy as np
import pandas as pd

from .data_loading import DataLoader


class Digitizer(DataLoader):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs):

	def apply_settings(self, settings):
		