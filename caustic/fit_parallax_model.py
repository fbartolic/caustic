import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import os, random
import sys
import exoplanet as xo
import theano.tensor as T

from data import OGLEData
from models import PointSourcePointLens
from models import PointSourcePointLensAnnualParallax
from models import PointSourcePointLensMarginalized
from models import OutlierRemovalModel
from utils import plot_map_model_and_residuals

random.seed(42)

# Load event data
data_dir = '/home/star/fb90/data/OGLE_ews/2017/blg-0525'
event = OGLEData(data_dir)

# Define output directories
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLens'
output_dir2 = 'output/' + event.event_name +\
        '/PointSourcePointLensAnnualParallax'

# Create output directory
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

# Plot data and save theplot
fig, ax = plt.subplots(figsize=(25, 10))
event.plot(ax)
plt.savefig('output/' + event.event_name + '/data.pdf')

#with PointSourcePointLens(event, 
#    errorbar_rescaling='additive_variance') as model:
#    model.sample(output_dir1, n_tune=500, n_samples=500)

with PointSourcePointLensAnnualParallax(event) as model:
    model.sample(output_dir2, n_tune=500, n_samples=500)