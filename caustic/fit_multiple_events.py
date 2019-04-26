import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os, random
import sys
import exoplanet as xo
import theano

from data import OGLEData
from models import PointSourcePointLens
from models import PointSourcePointLensMatern32
from models import PointSourcePointLensMarginalized

random.seed(42)

events = []  # data for each event

data_path = '/home/star/fb90/data/OGLE_ews/2017/'
dirs = []
for directory in os.listdir(data_path):
    dirs.append(directory)

random.shuffle(dirs)
i = 0
n_events = 200
for directory in dirs:
    if (i < n_events):
        event = OGLEData(data_path + directory)
        events.append(event)
        i = i + 1
     
for event in events:
    print("Fitting models for event ", event.event_name)

    # Define output directories
    output_dir1 = 'output/' + event.event_name +\
         '/PointSourcePointLens'
    output_dir2 = 'output/' + event.event_name +\
         '/PointSourcePointLensMarginalized'
    output_dir3 = 'output/' + event.event_name +\
         '/PointSourcePointLensMatern32'

    # Create output directory
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    if not os.path.exists(output_dir3):
        os.makedirs(output_dir3)

    # Plot data and save theplot
    fig, ax = plt.subplots(figsize=(25, 10))
    event.plot(ax)
    plt.savefig('output/' + event.event_name + '/data.pdf')

    # Fit models
    with PointSourcePointLens(event, 
            errorbar_rescaling='additive_variance') as model:
            model.sample(output_dir1, n_tune=500, n_samples=500)

    with PointSourcePointLensMarginalized(event, 
            errorbar_rescaling='additive_variance') as model:
            model.sample(output_dir2, n_tune=500, n_samples=500)

    with PointSourcePointLensMatern32(event, 
        errorbar_rescaling='additive_variance') as model:
        model.sample(output_dir3, n_tune=500, n_samples=500)