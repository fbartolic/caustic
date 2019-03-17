import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import os, random
import sys
import exoplanet as xo

from data import KMTData
from models import PointSourcePointLensMatern32
from models import PointSourcePointLensWhiteNoise1
from models import PointSourcePointLensWhiteNoise2
from models import PointSourcePointLensWhiteNoise3

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

    # I don't really understand how context managers worOGLEk
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

kmt_dir = '/home/star/fb90/data/KMT/kmtnet/2017/2017/KB170008'
event = KMTData(kmt_dir)
tables = event.get_standardized_data()
event.event_name = 'KMTKB170008'

# Remove outliers from data
mask1 = (tables[0]['flux'] < 4) & (tables[0]['flux']  > -1)
mask2 = (tables[1]['flux']  < 4) & (tables[1]['flux']  > -1)
mask3 = (tables[2]['flux']  < 4) & (tables[2]['flux']  > -1)

event.tables[0].remove_rows(np.argwhere(~mask1))
event.tables[1].remove_rows(np.argwhere(~mask2))
event.tables[2].remove_rows(np.argwhere(~mask3))

#event.tables = [event.tables[1]]

# Define output directories
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLensWN1'
output_dir2 = 'output/' + event.event_name +\
        '/PointSourcePointLensWN2'
output_dir3 = 'output/' + event.event_name +\
        '/PointSourcePointLensWN3'
output_dir4 = 'output/' + event.event_name +\
        '/PointSourcePointLensMatern32'

# Create output directory
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)
if not os.path.exists(output_dir4):
    os.makedirs(output_dir4)

# Plot data and save theplot
#fig, ax = plt.subplots(figsize=(25, 10))
#event.plot(ax)
#plt.savefig('output/' + event.event_name + '/data.pdf')

#fit_model(PointSourcePointLensWhiteNoise1(event), output_dir1)
#    fit_model(PointSourcePointLensWhiteNoise2(event), output_dir2)
#    fit_model(PointSourcePointLensWhiteNoise3(event), output_dir3)
fit_model(PointSourcePointLensMatern32(event), output_dir4)