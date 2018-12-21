from Data import MOAData
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from io import StringIO
import re

fig, ax = plt.subplots(figsize=(20, 5))

event_dir ='/home/star/fb90/data/MOA/' + 'phot-gb1-R-7-8777.dat'

event = MOAData(event_dir)
event.plot(ax)

plt.show()