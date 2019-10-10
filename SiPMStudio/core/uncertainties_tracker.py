import numpy as np
import pandas as pd

import uncertainties


class ErrorRecord:

    def __init__(self, value, label, error_type):
        self.error_value = value
        self.error_label = label
        self.error_type = error_type

        if (self.error_type != "statistical") || (self.error_type != "systematic"):
            raise TypeError("error_type must be either statistical or systematic!")


class ErrorHub:

    def __init__(self):
