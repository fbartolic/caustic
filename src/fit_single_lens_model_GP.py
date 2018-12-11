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
from exoplanet.gp import terms, GP
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from data_preprocessing_ogle import process_data
from plotting_utils import *
from SingleLensModelMatern32 import initialize_model

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
    model, gp = initialize_model(t, F, sigF)

    # Fit MAP model
    ##with model:
    #    map_soln = pm.find_MAP(start=model.test_point)

    t0_guess_idx = (np.abs(F - np.max(F))).argmin()
    # Initialization of the chain
    start = {'DeltaF':np.max(F), 'Fb':0., 'ln_K':-0.5, 'ln_sigma':-1., 
    'ln_rho':5., 'teff_tE':[35., 55.], 't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}
    
    # Sample model with NUTS
    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    with model:
        burnin = sampler.tune(tune=3000, start=start, 
            step_kwargs=dict(target_accept=0.95))
            
    with model:
        trace = sampler.sample(draws=2000)

    # Save stats summary to file
    if not os.path.exists('output/' + events[event_index]):
        os.makedirs('output/' + events[event_index])
    if not os.path.exists('output/' + events[event_index] + 
        '/PointSourcePointLensGP'):
        os.makedirs('output/' + events[event_index] + '/PointSourcePointLensGP')

    stats = pm.summary(trace)
    stats.to_csv('output/' + events[event_index] + '/PointSourcePointLensGP' +\
        '/summary.npy')

    # Save posterior samples
    samples_pymc3 = pm.trace_to_dataframe(trace, 
        varnames=["ln_sigma", "ln_rho", "ln_K","DeltaF", "Fb", "t0", 
            "teff_tE"]).values.T

    np.save('output/' + events[event_index] + '/PointSourcePointLensGP'  +\
        '/samples_pymc3.npy', samples_pymc3)

    # Save traceplots
    fig, ax = plt.subplots(8, 2 ,figsize=(20, 30))
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig('output/' + events[event_index] + '/PointSourcePointLensGP' +\
         '/traceplots_celerite_pymc3.png')

    pm.save_trace(trace, 'output/' + events[event_index] +\
         '/PointSourcePointLensGP' + '/samples.trace') 

    # Plot model in data space
    plt.clf()
    t_ = np.linspace(t[0], t[-1], 5000)

    # Generate 50 realizations of the prediction sampling randomly from the chain
    N_pred = 50
    pred_mu = np.empty((N_pred, len(t_)))
    pred_var = np.empty((N_pred, len(t_)))
    mean_function = np.empty((N_pred, len(t_)))
    with model:
        pred = gp.predict(t_, return_var=True)
        for i, sample in enumerate(get_samples_from_trace(trace, size=N_pred)):
            pred_mu[i], pred_var[i] = eval_in_model(pred, sample)
            mean_function[i] = eval_in_model(mean, sample)

    # Plot the predictions
    fig, ax = plt.subplots(figsize=(25, 6))
    for i in range(len(pred_mu)):
        mu = pred_mu[i]
        sd = np.sqrt(pred_var[i])
        label = None if i else "prediction"
        #art = plt.fill_between(t_, mu+sd, mu-sd, color="C1", alpha=0.1)
        #art.set_edgecolor("none")
        ax.plot(t_, mu - mean_function[i], color="C1", label=label, alpha=0.1)
        
    plot_data(ax, t, F, sigF) # Plot data
    plt.savefig('output/' + events[event_index] + '/PointSourcePointLensGP/' +\
        'model.png')

    # display the total number and percentage of divergent
    plt.clf()
    divergent = trace['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print('Percentage of Divergent %.1f' % divperc)

    pm.pairplot(trace,
            sub_varnames=['ln_sigma','ln_rho'],
            divergences=True,
            color='C3', figsize=(10, 5), kwargs_divergence={'color':'C2'})
    plt.savefig('output/' + events[event_index] + '/PointSourcePointLensGP' +\
         '/divergences.png')