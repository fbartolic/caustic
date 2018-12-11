import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
sys.path.append("../../exoplanet")
sys.path.append("codebase")

import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import exoplanet as xo
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from data_preprocessing_ogle import process_data
from plotting_utils import *
from SingleLensModel import initialize_model

mpl.rc('text', usetex=False)

events = [] # event names
lightcurves = [] # data for each event
 
i = 0
n_events = 5
data_path = '/home/star/fb90/data/OGLE_ews/2017'
for entry in sorted(os.listdir(data_path)):
    if (i < n_events):
        events.append(entry)
        print(entry)
        photometry = np.genfromtxt(data_path + '/' + entry + '/phot.dat', usecols=(0,1,2))
        lightcurves.append(photometry)
        i = i + 1
        
print("Loaded events:", events)

np.random.seed(42)

for event_index, lightcurve in enumerate(lightcurves):
    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

    # Fit pymc3 model
    model = initialize_model(t, F, sigF)

    t0_guess_idx = (np.abs(F - np.max(F))).argmin()
    # Initialization of the chain
    start = {'DeltaF':np.max(F), 'Fb':0.,
        't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}
    
    # Sample model with NUTS
    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    with model:
        burnin = sampler.tune(tune=3000, start=start, 
            step_kwargs=dict(target_accept=0.9))
            
    with model:
        trace = sampler.sample(draws=2000)

    # Save stats summary to file
    if not os.path.exists('output/' + events[event_index]):
        os.makedirs('output/' + events[event_index])
    if not os.path.exists('output/' + events[event_index] + 
        '/PointSourcePointLens'):
        os.makedirs('output/' + events[event_index] + '/PointSourcePointLens')

    stats = pm.summary(trace)
    stats.to_csv('output/' + events[event_index] + '/PointSourcePointLens' +\
        '/summary.npy')

    # Save posterior samples
    samples_pymc3 = pm.trace_to_dataframe(trace, 
        varnames=["ln_K","DeltaF", "Fb", "t0", 
            "teff_tE"]).values.T

    np.save('output/' + events[event_index] + '/PointSourcePointLens'  +\
        '/samples_pymc3.npy', samples_pymc3)

    # Save traceplots
    fig, ax = plt.subplots(6, 2 ,figsize=(30, 10))
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig('output/' + events[event_index] + '/PointSourcePointLens' +\
         '/traceplots_celerite_pymc3.png')

    ## display the total number and percentage of divergent
    #plt.clf()
    #divergent = trace['diverging']
    #print('Number of Divergent %d' % divergent.nonzero()[0].size)
    #divperc = divergent.nonzero()[0].size / len(trace) * 100
    #print('Percentage of Divergent %.1f' % divperc)

    #pm.pairplot(trace,
    #        sub_varnames=['ln_sigma','ln_rho'],
    #        divergences=True,
    #        color='C3', figsize=(30, 10), kwargs_divergence={'color':'C2'})
    #plt.savefig('output/' + events[event_index]+ '/PointSourcePointLens'  +\
    #     '/divergences.png')