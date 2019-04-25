import numpy as np
from data import KMTData
from utils import plot_model_and_residuals
from matplotlib import pyplot as plt
from models import PointSourcePointLens
from models import PointSourcePointLensMatern32
import pymc3 as pm
import os 

kmt_dir = '/home/star/fb90/data/KMT/kmtnet/2017/2017/KB170053'
event = KMTData(kmt_dir)
event.event_name = 'KMTKB170009'

# Remove worst outliers
event.remove_worst_outliers()
#event.tables = [event.tables[1]]
#event.masks = [event.masks[1]]

# Load traces
output_dir = 'output/KMTKB170053/PointSourcePointLensMatern32/'

# Plot non-GP models
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
    figsize=(25, 10), sharex=True)
fig.subplots_adjust(hspace=0.05)

plot_model_and_residuals(ax, event, 
    PointSourcePointLensMatern32(event), output_dir + 'model.trace', 50)

plt.show()

