import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os, random
import sys
import exoplanet as xo
import theano.tensor as T
import corner

from data import *
from astropy.table import Table
from astropy.coordinates import SkyCoord

from models import *
from utils import plot_histograms_of_prior_samples
from utils import plot_prior_model_samples
from utils import plot_model_and_residuals
from utils import plot_map_model_and_residuals

random.seed(42)

# Load event data
#data_dir = '/home/star/fb90/data/OGLE_ews/2017/blg-0441' # 0324(finite) 0441 (parallax)
#event = OGLEData(data_dir)

event_dir = 'data/OB05086/'
event = Data()
t = Table.read(event_dir + 'phot.dat', format='ascii', 
    names=('HJD', 'mag', 'mag_err'))
t['HJD'] = t['HJD'] + 2450000
t.meta = {'observatory':'OGLE', 'filter':'I'}
event.tables.append(t)
mask = np.ones(len(t['HJD']), dtype=bool)
event.masks.append(mask)
event.coordinates = SkyCoord("18h04m45.71s", "-26d59m15.2s")
event.event_name = 'OGLE-2005-BLG-086'
event.units = 'magnitudes'

# Define output directories
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLensAnnualParallax'
output_dir2 = 'output/' + event.event_name +\
        '/PointSourcePointLens'

# Create output directory
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)

if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)

# Plot data and save theplot
#fig, ax = plt.subplots(figsize=(25, 10))
#event.plot(ax);
#plt.savefig('output/' + event.event_name + '/data.pdf')

# Print library versions
print("Numpy version", np.__version__)
print("PyMC3 version", pm.__version__)

# Test non-GP model
with PointSourcePointLensAnnualParallax(event, kernel='white_noise', 
    errorbar_rescaling='none', 
    parametrization='angle_magnitude') as model1:
    print("Initializing model.")

with PointSourcePointLens(event, kernel='white_noise', 
    errorbar_rescaling='constant') as model2:
    print("Initializing model.")

# Sample the priors and save the plots
#plot_histograms_of_prior_samples(event, model1, output_dir1)

# Plot samples from prior in data space
#fig, ax = plt.subplots(figsize=(25, 10))
#plot_prior_model_samples(ax, event, model1, 100)
#plt.show()

#start = {'u0' : 0.37,
#    't0':2453628,
#     'tE':100,
#     'pi_EN':0.,
#     'pi_EE':0.}


# Find MAP solution and plot it
#with model1:
#    map_point = xo.optimize()
#
#print(map_point)
#
#fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#            figsize=(25, 10), sharex=True)
#
#plot_map_model_and_residuals(ax, event, model1, map_point)
#plt.show()

# Sample the model
with model1:
    print('model.vars' , model1.vars)
    trace1 = model1.sample(output_dir1, n_tune=2000, n_samples=1000)
#    trace1 = model1.sample_with_emcee(output_dir1, n_walkers=50, n_samples=5000)
#    trace1 = pm.load_trace(output_dir1 + '/model.trace')

# Plot corner plot of the samples
trace1_df = pm.trace_to_dataframe(trace1, include_transformed=True)
print('columns', trace1_df.columns)

params = ['Delta_F_lowerbound____0_0', 't0_interval__',
       'u0', 'tE_lowerbound__', 'pi_EE', 'pi_EN']

mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6

figure = corner.corner(trace1_df[params], quantiles=[0.16, 0.5, 0.84], 
    show_titles=True)
plt.show()

# Plot model
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
            figsize=(25, 10), sharex=True)
plot_model_and_residuals(ax, event, model1, output_dir1 + '/model.trace', 100)
plt.savefig(output_dir1 + '/model.png', bbox_inches='tight')
plt.show()

# Test GP model
#with PointSourcePointLens(event, kernel='matern32', 
#    errorbar_rescaling='additive_variance') as model2:
#    print("Initializing model.")

# Sample the priors and save the plots
#plot_histograms_of_prior_samples(event, model2, output_dir2)

# Plot samples from prior in data space
#fig, ax = plt.subplots(figsize=(25, 10))
#plot_prior_model_samples(ax, event, model2, 100)
#plt.show()

# Find MAP solution and plot it
#with model2:
#    map_point = xo.optimize()
#
#fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#            figsize=(25, 10), sharex=True)
#plot_map_model_and_residuals(ax, event, model2, map_point)
#plt.show()

# Sample the model
#with model2:
#    trace2 = model2.sample(output_dir2, n_tune=2000, n_samples=2000)
##    trace2 = pm.load_trace(output_dir1 + '/model.trace')
#
## Plot corner plot of the samples
#trace2_df = pm.trace_to_dataframe(trace2, include_transformed=True)
#print('columns', trace2_df.columns)
#
#params = ['Delta_F_lowerbound____0_0', 't0_interval__', 'teff_lowerbound__',
#       'u0_lowerbound__', 'sigma_lowerbound____0_0', 'rho_log____0_0']
#
#
#mpl.rcParams['axes.labelsize'] = 6
#mpl.rcParams['xtick.labelsize'] = 6
#mpl.rcParams['ytick.labelsize'] = 6
#mpl.rcParams['axes.titlesize'] = 6
#
#figure = corner.corner(trace2_df[params], quantiles=[0.16, 0.5, 0.84], 
#    show_titles=True)
#plt.show()
#
## Plot model
#fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#            figsize=(25, 10), sharex=True)
#plot_model_and_residuals(ax, event, model2, output_dir2 + '/model.trace', 100)
#plt.show()
