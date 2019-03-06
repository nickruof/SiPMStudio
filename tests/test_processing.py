import sys
sys.path.append("/Users/nickruof/Documents/LEGEND/SIPM/software_drafts/SiPMStudio/")
from SiPMStudio.core import data_loading
from SiPMStudio.core import devices
from SiPMStudio.core import digitizers

from SiPMStudio.processing import processor

path = "/Users/nickruof/Documents/LEGEND/SIPM/light_test.csv"
Digitizer = digitizers.CAENDT5730(df_data=path)
proc = processor.Processor()
proc.set_processor(Digitizer)

