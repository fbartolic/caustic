from Data import OGLEData
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from io import StringIO
import re

fig1, ax1 = plt.subplots(figsize=(20, 5))
fig, ax = plt.subplots(figsize=(20, 5))

event_dir ='../../../../data/OGLE_ews/2017/blg-0001' 
print(event_dir)

event = OGLEData(event_dir)
event.plot(ax1)
event.convert_data_to_fluxes()
event.plot_standardized_data(ax)
plt.show()