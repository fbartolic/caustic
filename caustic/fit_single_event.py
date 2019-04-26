import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import os, random
import sys
import exoplanet as xo
import theano.tensor as T

from data import KMTData
from models import PointSourcePointLens
from models import PointSourcePointLensMatern32
from models import PointSourcePointLensMarginalized
from models import OutlierRemovalModel
from utils import plot_map_model_and_residuals

random.seed(42)

# Load event data
kmt_dir = '/home/star/fb90/data/KMT/kmtnet/2017/2017/KB170035'
event = KMTData(kmt_dir)
event.event_name = 'KMTKB170035'


# Define output directories
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLens'
output_dir2 = 'output/' + event.event_name +\
        '/PointSourcePointLensMatern32'

# Create output directory
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

# Plot data and save theplot
fig, ax = plt.subplots(figsize=(25, 10))
event.plot(ax)
plt.savefig('output/' + event.event_name + '/data.pdf')

with PointSourcePointLens(event, 
    errorbar_rescaling='additive_variance') as model:
    model.sample(output_dir1, n_tune=500, n_samples=500)

with PointSourcePointLensMatern32(event, 
    errorbar_rescaling='additive_variance') as model:
    model.sample(output_dir2, n_tune=500, n_samples=500)