import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os, random
import sys
import exoplanet as xo
import theano.tensor as T
import corner

from data import OGLEData
from models import PointSourcePointLens
from models import FiniteSourcePointLens 
from models import PointSourcePointLensAnnualParallax
from models import PointSourcePointLensMarginalized
from models import OutlierRemovalModel
from utils import plot_histograms_of_prior_samples
from utils import plot_prior_model_samples
from utils import plot_model_and_residuals

random.seed(42)

# Load event data
data_dir = '/home/star/fb90/data/OGLE_ews/2017/blg-0441' # 0324(finite) 0441 (parallax)
event = OGLEData(data_dir)

# Define output directories
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLens'
output_dir2 = 'output/' + event.event_name +\
        '/FiniteSourcePointLens'
output_dir3 = 'output/' + event.event_name +\
        '/PointSourcePointLensAnnualParallax'

# Create output directory
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)

# Plot data and save theplot
#fig, ax = plt.subplots(figsize=(25, 10))
#event.plot(ax);
#plt.savefig('output/' + event.event_name + '/data.pdf')

# Print library versions
print("Numpy version", np.__version__)
print("PyMC3 version", pm.__version__)

# Sample the priors and save the plots
#plot_histograms_of_prior_samples(event, PointSourcePointLens(event, 
#    errorbar_rescaling='additive_variance'),  output_dir1)
#plot_histograms_of_prior_samples(event, FiniteSourcePointLens(event, 
#    errorbar_rescaling='additive_variance'),  output_dir2)
#plot_histograms_of_prior_samples(event, PointSourcePointLensAnnualParallax(event, 
#    errorbar_rescaling='additive_variance'),  output_dir3)

# Plot samples from prior in data space
#fig, ax = plt.subplots(figsize=(25, 10))
#plot_prior_model_samples(ax, event, PointSourcePointLensAnnualParallax(event, 
#    errorbar_rescaling='additive_variance'), 100)
#plt.show()

with PointSourcePointLens(event, 
    errorbar_rescaling='additive_variance') as model1:
    model1.sample(output_dir1, n_tune=2000, n_samples=2000)
#with FiniteSourcePointLens(event, 
#    errorbar_rescaling='additive_variance') as model2:
#    trace2 = model2.sample(output_dir2, n_tune=3000, n_samples=2000)
#    trace2 = pm.load_trace(output_dir2 + '/model.trace')
#with PointSourcePointLensAnnualParallax(event, 
#    errorbar_rescaling='additive_variance') as model3:
##    trace3 = model3.sample(output_dir3, n_tune=3000, n_samples=3000)
#    trace3 = pm.load_trace(output_dir3 + '/model.trace')

fig, ax = plt.subplots()
ax.hist(trace3['pi_E'])
# Plot scatterplot of samples in acceleration space
#fig, ax = plt.subplots()
#ax.scatter(trace3['a_vert'], trace3['a_par'], color='black', marker='o', 
#    alpha=0.05)
#ax.set_xlabel('a_vert')
#ax.set_ylabel('a_par')
#ax.grid()
plt.show()
#
## Plot corner plot of the samples
#trace3_df = pm.trace_to_dataframe(trace3, include_transformed=True)
#print('columns', trace3_df.columns)
#
#params = ['Delta_F_lowerbound____0_0', 't0_prime_interval__',
#       'u0_prime', 'omega_E_prime_lowerbound__', 'a_par', 'a_vert']
#
#mpl.rcParams['axes.labelsize'] = 6
#mpl.rcParams['xtick.labelsize'] = 6
#mpl.rcParams['ytick.labelsize'] = 6
#mpl.rcParams['axes.titlesize'] = 6
#
#figure = corner.corner(trace3_df[params], quantiles=[0.16, 0.5, 0.84], 
#    show_titles=True)
#plt.show()

# Plot model PSPL model
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
            figsize=(25, 10), sharex=True)
plot_model_and_residuals(ax, event, 
            PointSourcePointLens(event, errorbar_rescaling='additive_variance'),
            output_dir1 + '/model.trace', 100)
plt.show()

# Plot model parallax model
#fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#            figsize=(25, 10), sharex=True)
#plot_model_and_residuals(ax, event, 
#            PointSourcePointLensAnnualParallax(event, errorbar_rescaling='additive_variance'),
#            output_dir3 + '/model.trace', 100)
#plt.show()