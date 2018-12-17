import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
sys.path.append("../../exoplanet")
sys.path.append("models")
sys.path.append("codebase")

import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import exoplanet as xo
from exoplanet.gp import terms, GP

from data_preprocessing_ogle import process_data
from plotting_utils import *
from SingleLensModels import ZeroMeanMatern32StudentT

mpl.rc('text', usetex=False)

events = [] # event names
lightcurves = [] # data for each event
limits = [170, 1900, 1900, 1700, 2500]
 
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

    t = t[:limits[event_index]]
    F = F[:limits[event_index]]
    sigF = sigF[:limits[event_index]]


    # Save processed data
    if not os.path.exists('output/' + events[event_index]):
        os.makedirs('output/' + events[event_index])
    
    # Fit a GP model
    model = ZeroMeanMatern32StudentT(t, F, sigF)

    # Sample models with NUTS
    start_GP = {'sigma':0.5, 'rho':np.exp(2.)}
    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    # Sample GP model
    with model as model_matern32:
        trace_gp = pm.sample(2000, tune=1000, 
            nuts_kwargs=dict(target_accept=0.9))

    #with model_matern32:
    #    for RV in model_matern32.basic_RVs:
    #        print(RV.name, RV.logp(model_matern32.test_point)) 

    #with model_matern32:
    #    burnin = sampler.tune(tune=3000, start=start_GP, 
    #        step_kwargs=dict(target_accept=0.9), chains=1)
    #        
    #with model_matern32:
    #    trace_gp = sampler.sample(draws=2000)

    # Save output stats to file
    output_dir_gp = 'output/' + events[event_index] +\
        '/ZeroMeanMatern32StudentT'
    if not os.path.exists(output_dir_gp):
        os.makedirs(output_dir_gp)

    def save_summary_stats(trace, output_dir):
        df = pm.summary(trace) 
        df = df.round(2)
        df.to_csv(output_dir + '/sampling_stats.csv')

    save_summary_stats(trace_gp, output_dir_gp)

    # Save posterior samples
    pm.save_trace(trace_gp, output_dir_gp + '/model.trace', overwrite=True)

    # Save traceplots
    def save_traceplots(trace, n_pars, output_dir):
        fig, ax = plt.subplots(n_pars, 2 ,figsize=(20, 30))
        _ = pm.traceplot(trace, ax=ax)
        plt.savefig(output_dir + '/traceplots.png')

    save_traceplots(trace_gp, 11, output_dir_gp)

    # Display the total number and percentage of divergent
    def save_divergences_stats(trace, output_dir):
        with open(output_dir + "/divergences.txt", "w") as text_file:
            divergent = trace['diverging']
            print(f'Number of Divergent %d' % divergent.nonzero()[0].size, 
                file=text_file)
            divperc = divergent.nonzero()[0].size / len(trace) * 100
            print(f'Percentage of Divergent %.1f' % divperc, file=text_file)

    save_divergences_stats(trace_gp, output_dir_gp)