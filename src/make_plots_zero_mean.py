import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
sys.path.append("../../exoplanet")
sys.path.append("codebase")
sys.path.append("models")

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
from SingleLensModels import ZeroMeanMatern32

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16

events = [] # event names
lightcurves = [] # data for each event

# Iterate over events, load data, and make plots 
for event_index, entry in enumerate(os.scandir('output')):
    if entry.is_dir():
        data = np.load(entry.path + '/data_zero_mean.npy')
        t, F, sigF = data[:, 0], data[:, 1], data[:, 2]

        # Set up GP model
        model = ZeroMeanMatern32(t, F, sigF)
        with model as model_matern32:
            trace = pm.load_trace(entry.path + '/ZeroMeanMatern32/' +\
            'model.trace') 

        t_ = np.linspace(t[0], t[-1], 2000)

        # Generate 50 realizations of the prediction sampling randomly from the chain
        N_pred = 50
        pred_mu = np.empty((N_pred, len(t_)))
        pred_var = np.empty((N_pred, len(t_)))

        with model_matern32:
            pred = model_matern32.gp.predict(t_, return_var=True)
            for i, sample in enumerate(xo.get_samples_from_trace(trace, size=N_pred)):
               pred_mu[i], pred_var[i] = xo.eval_in_model(pred, sample)

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(25, 6))
        for i in range(len(pred_mu)):
            mu = pred_mu[i]
            sd = np.sqrt(pred_var[i])
            label = None if i else "prediction"
            art = plt.fill_between(t_, mu+sd, mu-sd, color="C1", alpha=0.1)
            art.set_edgecolor("none")
            #ax.plot(t_, mean_function[i] - mu, color="C1", label=label, alpha=0.1)
            
        plot_data(ax, t, F, sigF) # Plot data

        plt.savefig(entry.path + '/ZeroMeanMatern32/' +\
            'model.png')