import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os, random
import sys
import exoplanet as xo
import theano
import corner

from data import OGLEData
from models import PointSourcePointLens
from models import FiniteSourcePointLens
from utils import plot_model_and_residuals

random.seed(42)

events = []  # data for each event

data_path = '/home/star/fb90/data/OGLE_ews/2017/'
dirs = []
for directory in os.listdir(data_path):
    dirs.append(directory)

random.shuffle(dirs)
#i = 0
#n_events = 200
#for directory in dirs:
#    if (i < n_events):
#        event = OGLEData(data_path + directory)
#        events.append(event)
#        i = i + 1

event = OGLEData(data_path + 'blg-0627')
events.append(event)
     
for event in events:
    print("Fitting models for event ", event.event_name)

    # Define output directories
    output_dir1 = 'output/' + event.event_name +\
         '/PointSourcePointLens/'

    # Create output directory
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)

    # Plot data and save theplot
    fig, ax = plt.subplots(figsize=(25, 10))
    event.plot(ax)
    plt.savefig('output/' + event.event_name + '/data.pdf')

    # Fit models
    with PointSourcePointLens(event, kernel='white_noise', 
        errorbar_rescaling='additive_variance') as model1:
        print("Initializing model.")

    # Sample the model
    with model1:
#        trace1 = model1.sample(output_dir1, n_tune=4000, n_samples=4000)
        trace1 = pm.load_trace(output_dir1 + '/model.trace')

    # Plot corner plot of the samples
#    trace1_df = pm.trace_to_dataframe(trace1, include_transformed=True)
    #print('columns', trace1_df.columns)

#    params = ['Delta_F_lowerbound____0_0', 't0_interval__',
#        'u0_lowerbound__', 'teff_lowerbound__'] #'rho_star_lowerbound__']
#
#    mpl.rcParams['axes.labelsize'] = 10
#    mpl.rcParams['xtick.labelsize'] = 10
#    mpl.rcParams['ytick.labelsize'] = 12
#    mpl.rcParams['axes.titlesize'] = 8
#
#    figure = corner.corner(trace1_df[params], quantiles=[0.16, 0.5, 0.84], 
#        show_titles=True)
#    plt.savefig(output_dir1 + '/corner.png', bbox_inches='tight')

    # Plot model
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
                figsize=(12, 5), sharex=True)
    plot_model_and_residuals(ax, event, model1, output_dir1 + '/model.trace', 100)

    plt.savefig(output_dir1 + '/model.png', bbox_inches='tight')