import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import os, random
import sys
import exoplanet as xo

from data import KMTData
from models import PointSourcePointLens
from models import PointSourcePointLensMatern32
from models import PointSourcePointLensMarginalized
from models import OutlierRemovalModel
from utils import plot_map_model_and_residuals

def save_summary_stats(trace, output_dir):
        df = pm.summary(trace)
        df = df.round(2)
        df.to_csv(output_dir + '/sampling_stats.csv')

def save_divergences_stats(trace, output_dir):
    with open(output_dir + "/divergences.txt", "w") as text_file:
        divergent = trace['diverging']
        print(f'Number of Divergent %d' % divergent.nonzero()[0].size,
                file=text_file)
        divperc = divergent.nonzero()[0].size / len(trace) * 100
        print(f'Percentage of Divergent %.1f' % divperc, file=text_file)

def fit_model(model, output_dir, n_tune=2000, n_sample=2000):

    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    # I don't really understand how context managers work
    with model as model_instance:
        print("Free parameters: ", model.free_parameters)
        print("Initial values of logp for each parameter: ", 
            model.initial_logps)
        
    with model_instance:
        burnin = sampler.tune(tune=n_tune,
            step_kwargs=dict(target_accept=0.9))

    with model_instance:
        trace = sampler.sample(draws=n_sample)

    # Save trace to file
    pm.save_trace(trace, output_dir + '/model.trace',
        overwrite=True)
    df = pm.trace_to_dataframe(trace) 
    df.to_csv(output_dir + '/trace.csv',)
    
    # Save output stats to file
    save_summary_stats(trace, output_dir)

    # Save stats about divergent samples 
    save_divergences_stats(trace, output_dir)

    # Save traceplots
    _ = pm.traceplot(trace)
    plt.savefig(output_dir + '/traceplots.png')

    # Save autocorrelation plots for the chains
    pm.plots.autocorrplot(trace)
    plt.savefig(output_dir + '/autocorr.png')

    # Save corner plot of the samples
    rvs = [rv.name for rv in model_instance.basic_RVs]
    pm.pairplot(trace,
                divergences=True, plot_transformed=True, text_size=25,
                varnames=rvs,
                color='C3', figsize=(40, 40), kwargs_divergence={'color': 'C0'})
    plt.savefig(output_dir + '/pairplot.png')

random.seed(42)

kmt_dir = '/home/star/fb90/data/KMT/kmtnet/2017/2017/KB170053'
event = KMTData(kmt_dir)
event.event_name = 'KMTKB170053'

# Remove worst outliers
event.remove_worst_outliers()
#event.tables = [event.tables[1]]
#event.masks = [event.masks[1]]

# Plot data
fig, ax = plt.subplots(figsize=(25, 10))
event.plot_standardized_data(ax)

# Optimize the GP model to remove outliers
#with PointSourcePointLensWhiteNoise3(event) as model:
#    start = model.test_point
#    map_soln = xo.optimize(start=start, vars=[model.F_base])
#    map_soln = xo.optimize(start=map_soln, vars=[model.Delta_F])
#    map_soln = xo.optimize(start=map_soln, vars=[model.u0])
#    map_soln = xo.optimize(start=map_soln, vars=[model.t0])
#    map_soln = xo.optimize(start=map_soln, vars=[model.t0, model.teff])
#    map_soln = xo.optimize(start=map_soln)
#
#print(map_soln)
#
#fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#    figsize=(25, 10), sharex=True)
#fig.subplots_adjust(hspace=0.05)
#
#plot_map_model_and_residuals(ax, event, model, map_soln)
#plt.show()

# Define output directories
output_dir4 = 'output/' + event.event_name +\
        '/PointSourcePointLens'

# Create output directory
if not os.path.exists(output_dir4):
    os.makedirs(output_dir4)

# Plot data and save theplot
#fig, ax = plt.subplots(figsize=(25, 10))
#event.plot(ax)
#plt.savefig('output/' + event.event_name + '/data.pdf')

#fit_model(PointSourcePointLensWhiteNoise1(event), output_dir1)
#    fit_model(PointSourcePointLensWhiteNoise2(event), output_dir2)
#    fit_model(PointSourcePointLensWhiteNoise3(event), output_dir3)

# Profile model
#with PointSourcePointLensMarginalized(event) as model_instance:
#    model_instance.profile(model_instance.logpt).summary()


fit_model(PointSourcePointLens(event), output_dir4, 1000, 1000)