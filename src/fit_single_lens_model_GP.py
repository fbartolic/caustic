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

from Data import OGLEData
from SingleLensModels import PointSourcePointLens
from SingleLensModels import PointSourcePointLensMatern32

mpl.rc('text', usetex=False)

events = [] # data for each event
 
i = 0
n_events = 100
data_path = '/home/star/fb90/data/OGLE_ews/2017/'
for entry in sorted(os.listdir(data_path)):
    if (i < n_events):
        event = OGLEData(data_path + entry)
        events.append(event)
        i = i + 1

np.random.seed(42)

for event in events: 
    print("Fitting models for event ", event.event_name)

    # Fit a non GP and a GP model
    model1 = PointSourcePointLensMatern32(event, parametrization='other')
    model2 = PointSourcePointLens(event, parametrization='other')

    # Sample models with NUTS
    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    # Sample non-GP model
    with model2 as model_standard:
        for RV in model_standard.basic_RVs:
            print(RV.name, RV.logp(model_standard.test_point))

    with model_standard:
        burnin = sampler.tune(tune=3000, 
            step_kwargs=dict(target_accept=0.95))
            
    with model_standard:
        trace_standard = sampler.sample(draws=2000)

    # Sample GP model
    with model1 as model_matern32:
        for RV in model_matern32.basic_RVs:
            print(RV.name, RV.logp(model_matern32.test_point))

    # start at the mean posterior values of non-GP model
    print(trace_standard['ln_teff_ln_tE'][0].mean())
    start_GP = {
        'DeltaF': trace_standard['DeltaF'].mean(),
        'Fb': trace_standard['Fb'].mean(),
        't0': trace_standard['t0'].mean(),
        'ln_teff_ln_tE': [trace_standard['ln_teff_ln_tE'][0].mean(),
            trace_standard['ln_teff_ln_tE'][1].mean()],
        'K': trace_standard['K'].mean()
    }
    with model_matern32:
        burnin = sampler.tune(tune=3000, start=start_GP, 
            step_kwargs=dict(target_accept=0.95))
            
    with model_matern32:
        trace_gp = sampler.sample(draws=2000)

    # Save output stats to file
    output_dir_standard = 'output/' + event.event_name + '/PointSourcePointLens'
    output_dir_gp = 'output/' + event.event_name + '/PointSourcePointLensGP'
    if not os.path.exists(output_dir_standard):
        os.makedirs(output_dir_standard)
    if not os.path.exists(output_dir_gp):
        os.makedirs(output_dir_gp)

    def save_summary_stats(trace, output_dir):
        df = pm.summary(trace) 
        df = df.round(2)
        df.to_csv(output_dir + '/sampling_stats.csv')

    save_summary_stats(trace_standard, output_dir_standard)
    save_summary_stats(trace_gp, output_dir_gp)

    # Save posterior samples
    pm.save_trace(trace_standard, output_dir_standard + '/model.trace', 
        overwrite=True)
    pm.save_trace(trace_gp, output_dir_gp + '/model.trace', overwrite=True)

    # Save traceplots
    def save_traceplots(trace, n_pars, output_dir):
        fig, ax = plt.subplots(n_pars, 2 ,figsize=(20, 30))
        _ = pm.traceplot(trace, ax=ax)
        plt.savefig(output_dir + '/traceplots.png')

    save_traceplots(trace_standard, 7, output_dir_standard)
    save_traceplots(trace_gp, 11, output_dir_gp)

    # Display the total number and percentage of divergent
    def save_divergences_stats(trace, output_dir):
        with open(output_dir + "/divergences.txt", "w") as text_file:
            divergent = trace['diverging']
            print(f'Number of Divergent %d' % divergent.nonzero()[0].size, 
                file=text_file)
            divperc = divergent.nonzero()[0].size / len(trace) * 100
            print(f'Percentage of Divergent %.1f' % divperc, file=text_file)

    save_divergences_stats(trace_standard, output_dir_standard)
    save_divergences_stats(trace_gp, output_dir_gp)

    pm.pairplot(trace_standard,
            divergences=True, plot_transformed=True, text_size=11,
            color='C3', figsize=(40, 40), kwargs_divergence={'color':'C0'})
    plt.savefig(output_dir_standard + '/pairplot.png')

    pm.pairplot(trace_gp,
            divergences=True, plot_transformed=True, text_size=11,
            color='C3', figsize=(40, 40), kwargs_divergence={'color':'C0'})
    plt.savefig(output_dir_gp + '/pairplot.png')