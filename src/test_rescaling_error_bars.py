from theano.ifelse import ifelse
import theano.tensor as T
import theano
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os, random
import sys

sys.path.append("../../exoplanet")
sys.path.append("models")
sys.path.append("codebase")
from SingleLensModels import PointSourcePointLensRescaling
from Data import OGLEData
import exoplanet as xo

mpl.rc('text', usetex=False)
random.seed(42)

events = []  # data for each event

data_path = '/home/star/fb90/data/OGLE_ews/2017/'
dirs = []
for directory in os.listdir(data_path):
    dirs.append(directory)

random.shuffle(dirs)
i = 0
n_events = 20
for directory in dirs:
    if (i < n_events):
        event = OGLEData(data_path + directory)
        events.append(event)
        i = i + 1

#event = OGLEData('/home/star/fb90/data/OGLE_ews/2017/blg-1403')
#events.append(event) 

for event in events:
    print("Fitting models for event ", event.event_name)

    # Define output directories
    output_dir_standard = 'output/' + event.event_name + '/PointSourcePointLensRescaling'
    if not os.path.exists(output_dir_standard):
        os.makedirs(output_dir_standard)

    # Fit a non GP and a GP model
    model1 = PointSourcePointLensRescaling(event, use_joint_prior=True)

    # Sample models with NUTS
    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    # Sample non-GP model
    print("Sampling model without GP:")
    with model1 as model_standard:
        for RV in model_standard.basic_RVs:
            print(RV.name, RV.logp(model_standard.test_point))

    with model_standard:
        burnin = sampler.tune(tune=3000,
                              step_kwargs=dict(target_accept=0.95))

    with model_standard:
        trace_standard = sampler.sample(draws=2000)


    # Save posterior samples
    pm.save_trace(trace_standard, output_dir_standard + '/model.trace',
        overwrite=True)

    # Save output stats to file
    def save_summary_stats(trace, output_dir):
        df = pm.summary(trace)
        df = df.round(2)
        df.to_csv(output_dir + '/sampling_stats.csv')

    save_summary_stats(trace_standard, output_dir_standard)

    # Display the total number and percentage of divergent
    def save_divergences_stats(trace, output_dir):
        with open(output_dir + "/divergences.txt", "w") as text_file:
            divergent = trace['diverging']
            print(f'Number of Divergent %d' % divergent.nonzero()[0].size,
                  file=text_file)
            divperc = divergent.nonzero()[0].size / len(trace) * 100
            print(f'Percentage of Divergent %.1f' % divperc, file=text_file)

    save_divergences_stats(trace_standard, output_dir_standard)

    rvs = [rv.name for rv in model_standard.basic_RVs]
    pm.pairplot(trace_standard,
                divergences=True, plot_transformed=True, text_size=25,
                varnames=rvs[:-1],
                color='C3', figsize=(40, 40), kwargs_divergence={'color': 'C0'})
    plt.savefig(output_dir_standard + '/pairplot.png')