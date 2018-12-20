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
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from Data import OGLEData
from SingleLensModels import PointSourcePointLens


from simulation_based_calibration import SBC

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


    output_dir_standard = 'output/' + event.event_name + '/PointSourcePointLens'

    # Fit a non GP and a GP model
    model1 = PointSourcePointLens(event)

#    # Sample prior predictive distribution
#    t_ = np.linspace(event.df['HJD - 2450000'].values[0], 
#        event.df['HJD - 2450000'].values[-1], 1000)
#
#    with model1 as model_standard:
#        trace = pm.sample_prior_predictive(50)
#
#    Fb = trace['Fb']
#    u0 = trace['u0']
#    t0 = trace['t0']
#    DeltaF = trace['DeltaF']
#    tE = trace['tE']
#
#    # Plot the predictions
#    fig, ax = plt.subplots(figsize=(25, 6))
#    for i in range(len(Fb)):
#        u = np.sqrt(u0[i]**2 + ((t_ - t0[i])/tE[i])**2)
#        A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))
#        mean_func = DeltaF[i]*(A(u) - 1)/(A(u0[i]) - 1) + Fb[i]
#        ax.plot(t_, mean_func, color="C1", alpha=0.5)
#
#    plt.savefig(output_dir_standard + '/prior_predictive.png')

