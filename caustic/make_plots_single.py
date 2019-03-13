import numpy as np
from data import KMTData
from utils import plot_model_and_residuals
from matplotlib import pyplot as plt
from models import PointSourcePointLensWhiteNoise1
from models import PointSourcePointLensMatern32
import pymc3 as pm
import os 

kmt_dir = '/home/star/fb90/data/KMT/kmtnet/2017/2017/KB170008'
event = KMTData(kmt_dir)
event.event_name = 'KMTKB170008'

tables = event.get_standardized_data()

# Remove outliers from data
mask1 = (tables[0]['flux'] < 4) & (tables[0]['flux']  > -1)
mask2 = (tables[1]['flux']  < 4) & (tables[1]['flux']  > -1)
mask3 = (tables[2]['flux']  < 4) & (tables[2]['flux']  > -1)

event.tables[0].remove_rows(np.argwhere(~mask1))
event.tables[1].remove_rows(np.argwhere(~mask2))
event.tables[2].remove_rows(np.argwhere(~mask3))

# Load trace
output_dir = 'output/' + event.event_name + '/PointSourcePointLensWN1/' 

# Plot non-GP models
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
    figsize=(25, 10), sharex=True)
fig.subplots_adjust(hspace=0.05)


plot_model_and_residuals(ax, event, 
    PointSourcePointLensWhiteNoise1(event), output_dir + 'model.trace')
plt.savefig(output_dir + 'model.png')

plt.show()
