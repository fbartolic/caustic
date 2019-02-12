import os, random
import sys
from matplotlib import pyplot as plt
sys.path.append("models")
sys.path.append("codebase")
from Data import OGLEData

path_to_ogle_event_dir = '/home/fran/data/OGLE_ews/2017/blg-0001'
event = OGLEData(path_to_ogle_event_dir)
print(event.df)
fig, ax = plt.subplots()
event.plot(ax)
plt.show()

