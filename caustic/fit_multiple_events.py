import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os, random
import sys
import exoplanet as xo
import theano

from data import OGLEData
from models import PointSourcePointLens
from models import PointSourcePointLensMatern32
from models import PointSourcePointLensMarginalized

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

def fit_model(model, output_dir, n_tune=2000, n_samples=2000):

    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    # I don't really understand how context managers worOGLEk
    with model as model_instance:
        print("Free parameters: ", model.free_parameters)
        print("Free parameter values: ", model.test_point)
        print("Initial values of logp for each parameter: ", 
            model.initial_logps)

    with model_instance:
        burnin = sampler.tune(tune=n_tune,
            step_kwargs=dict(target_accept=0.9))

    with model_instance:
        trace = sampler.sample(draws=n_samples)

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

events = []  # data for each event

data_path = '/home/star/fb90/data/OGLE_ews/2017/'
dirs = []
for directory in os.listdir(data_path):
    dirs.append(directory)

random.shuffle(dirs)
i = 0
n_events = 200
for directory in dirs:
    if (i < n_events):
        event = OGLEData(data_path + directory)
        events.append(event)
        i = i + 1
     
for event in events:
    print("Fitting models for event ", event.event_name)

    # Define output directories
    output_dir1 = 'output/' + event.event_name +\
         '/PointSourcePointLensMatern32'
    output_dir2 = 'output/' + event.event_name +\
         '/PointSourcePointLensMarginalized'

    # Create output directory
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)

    # Plot data and save theplot
    fig, ax = plt.subplots(figsize=(25, 10))
    event.plot(ax)
    plt.savefig('output/' + event.event_name + '/data.pdf')

#    with PointSourcePointLens(event) as model_instance:
#        model_instance.profile(model_instance.logpt).summary()
#    with PointSourcePointLensMarginalized(event) as model_instance:
#        model_instance.profile(model_instance.logpt).summary()

#    fit_model(PointSourcePointLensMatern32(event, 
#        errorbar_rescaling='additive_variance'), output_dir1)

    fit_model(PointSourcePointLensMarginalized(event, 
        errorbar_rescaling='additive_variance'), output_dir2, n_samples=1000)