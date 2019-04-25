import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import os, random
import sys
import exoplanet as xo
import theano.tensor as T

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

def remove_outliers(model, event):
    event.remove_worst_outliers()
    with model as model_instance:
        start = model.test_point
        map_soln = xo.optimize(start=start, vars=[model.F_base])
        map_soln = xo.optimize(start=map_soln, vars=[model.Delta_F])
        map_soln = xo.optimize(start=map_soln, vars=[model.u0])
        map_soln = xo.optimize(start=map_soln, vars=[model.t0])
        map_soln = xo.optimize(start=map_soln, vars=[model.u0, model.teff])
        map_soln = xo.optimize(start=map_soln)

        t_observed = [T._shared(table['HJD']) for table in event.tables]
        pred_at_observed_times = model.evaluate_map_model_on_grid(
            t_observed, map_soln)

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
        figsize=(25, 10), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    
    plot_map_model_and_residuals(ax, event, model, map_soln)

    for n in range(model.n_bands):
        resid = event.tables[n]['flux'] - pred_at_observed_times[n]
        rms = np.sqrt(np.median(resid**2))

        # Updata mask
        mask = np.abs(resid) < 7*rms
        event.masks[n] = mask

    fig, ax = plt.subplots(figsize=(25, 10))
    event.plot_standardized_data(ax)


random.seed(42)

# Load event data
kmt_dir = '/home/star/fb90/data/KMT/kmtnet/2017/2017/KB170053'
event = KMTData(kmt_dir)
event.event_name = 'KMTKB170053'

# Remove outliers
remove_outliers(OutlierRemovalModel(event), event)

plt.show()

# Optimize the GP model to remove outliers
#with OutlierRemovalModel(event) as model:
#    start = model.test_point
#    map_soln = xo.optimize(start=start, vars=[model.F_base])
#    map_soln = xo.optimize(start=map_soln, vars=[model.Delta_F])
#    map_soln = xo.optimize(start=map_soln, vars=[model.u0])
#    map_soln = xo.optimize(start=map_soln, vars=[model.t0])
#    map_soln = xo.optimize(start=map_soln, vars=[model.u0, model.teff])
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
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLens'
output_dir2 = 'output/' + event.event_name +\
        '/PointSourcePointLensMatern32'

# Create output directory
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

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


#fit_model(PointSourcePointLens(event), output_dir1, 500, 1000)
#fit_model(PointSourcePointLensMatern32(event), output_dir2, 2000, 3000)