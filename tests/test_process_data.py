import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")
from SiPMStudio.core import data_loading
from SiPMStudio.core import devices
from SiPMStudio.core import digitizers

from SiPMStudio.processing import processor