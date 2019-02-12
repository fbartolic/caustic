import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import os
import sys
sys.path.append("codebase")
sys.path.append("models")

import pymc3 as pm
import arviz as az
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import exoplanet as xo
from exoplanet.gp import terms, GP
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from Data import OGLEData
from SingleLensModels import PointSourcePointLens

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16

def qq_plot(ax, residuals):
    percentiles = np.linspace(0, 100, 500)
    quantile_residuals = np.percentile(residuals, percentiles)
    quantile_model = np.percentile(np.random.normal(size=10000), percentiles)

    ax.scatter(quantile_model, quantile_residuals, marker='o', color='black',
        alpha=0.3)
    ax.grid()
    x = np.linspace(quantile_model[0], quantile_model[-1])
    ax.plot(x, x, color='black', linestyle='dashed', alpha=0.8)
    ax.set_aspect(1)
    ax.set_xlabel("Modeled residuals")
    ax.set_ylabel("Measured residuals")

def evaluate_model_on_grid(data, pm_model, trace, t_grid, N_pred):
    # Generate 50 realizations of the prediction sampling randomly from the chain
    mean_function = np.empty((N_pred, len(t_grid)))

    with pm_model:
        # Evaluate model
        for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                size=N_pred)):
            Fb = pm_model.Fb
            u0 = pm_model.u0
            t0 = pm_model.t0
            DeltaF = pm_model.DeltaF
            tE = pm_model.tE
            u = T.sqrt(u0**2 + ((t_grid - t0)/tE)**2)
            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
            mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

            mean_function[i] = xo.eval_in_model(mean_func, sample)

    return mean_function 
        
def evaluate_median_model_on_grid(data, pm_model, trace, t_grid):
    median_params_dict = {}
    for i in range(len(pm_model.vars)):
        median_params_dict[str(pm_model.vars[i])] = \
            np.percentile(trace[pm_model.vars[i]], [16, 50, 84], 
            axis=0)[1]

    with pm_model:
        # Evaluate residuals
        Fb = pm_model.Fb
        u0 = pm_model.u0
        t0 = pm_model.t0
        DeltaF = pm_model.DeltaF
        tE = pm_model.tE
        u = T.sqrt(u0**2 + ((t_grid - t0)/tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
        mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

        mean_function = xo.eval_in_model(mean_func, 
            median_params_dict)

    return mean_function 

def plot_mock_model(ax, data, pm_model, trace):
    median_params_dict = {}
    for i in range(len(pm_model.vars)):
        median_params_dict[str(pm_model.vars[i])] = \
            np.percentile(trace[pm_model.vars[i]], [16, 50, 84], 
            axis=0)[1]
    
    df = event.get_standardized_data()
    t = df['HJD - 2450000'].values

    with pm_model:
        # Evaluate residuals
        Fb = pm_model.Fb
        u0 = pm_model.u0
        t0 = pm_model.t0
        DeltaF = pm_model.DeltaF
        tE = pm_model.tE
        u = T.sqrt(u0**2 + ((t - t0)/tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
        mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

        mean_function = xo.eval_in_model(mean_func, 
            median_params_dict)
        
    F_err = df['I_flux_err'].values*np.median(trace['K'])

    mock_data = np.random.multivariate_normal(mean=mean_function,
        cov=np.diag(F_err))

    ax[0].errorbar(t, mock_data, F_err, fmt='.', color='black', label='Data', 
        ecolor='#686868')
    ax[0].grid(True)
    ax[0].set_title(data.event_name + 'mock')
    ax[0].set_xlabel(df.columns[0])
    ax[0].set_ylabel(df.columns[1])

    # Plot residuals
    df = event.get_standardized_data()
    residuals = mock_data - mean_function
    ax[1].errorbar(t, residuals, F_err, 
        fmt='.', color='black', ecolor='#686868')
    ax[1].grid(True)

def plot_model_and_residuals(ax, event, pm_model, trace, ti_idx,
    tf_idx):
    N_pred = 50

    df = event.get_standardized_data()
    time = df['HJD - 2450000'].values

    t_grid = np.linspace(time[ti_idx], time[tf_idx], 1000)
    t = time[ti_idx:tf_idx]

    predictions = evaluate_model_on_grid(event, pm_model, trace,
        t_grid, N_pred)
    
    mask = ((time >= time[ti_idx]) & (time <= time[tf_idx]))
    event.plot_standardized_data(ax[0], mask)
    ax[0].set_xlabel(None)

    for i in range(N_pred):
        ax[0].plot(t_grid, predictions[i, :], color='C1', alpha=0.2)

    # Plot residuals
    df = event.get_standardized_data()
    residuals = df['I_flux'].values[ti_idx:tf_idx] -\
        evaluate_median_model_on_grid(event, pm_model, trace, t)
    ax[1].errorbar(t, residuals, df['I_flux_err'].values[ti_idx:tf_idx], 
        fmt='.', color='black', ecolor='#686868')
    ax[1].grid(True)

def plot_traceplots(trace, n_pars, output_dir):
    fig, ax = plt.subplots(n_pars, 2, figsize=(20, 30))
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig(output_dir + '/traceplots.png')

events = [] # event names
lightcurves = [] # data for each event
data_path = '/home/fran/data/OGLE_ews/2017/'

DeltaF_medians = []
Fb_medians = []
t0_medians = []
u0_medians = []
tE_medians = []
K_medians = []

# Iterate over events, load data, and make plots 
for entry in os.scandir('output'):
    if entry.is_dir():
        data_dir = data_path + 'blg-' + entry.name[-4:]
        event = OGLEData(data_dir)
        print(event.event_name)

        # Load traces
        output_dir = 'output/' + event.event_name + '/PointSourcePointLens/' 
        model_standard = PointSourcePointLens(event, use_joint_prior=False)
        
        with model_standard:
            trace_standard = pm.load_trace(output_dir + 'model.trace') 

        # Plot traceplots
        #plot_traceplots(trace_standard, 15, output_dir)

        # Plot model
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
                figsize=(25, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)
        plot_model_and_residuals(ax, event, model_standard, trace_standard, 0,
            len(event.df['HJD - 2450000']) - 1)
        plt.savefig(output_dir + 'model.png')

        # Trace in dataframe format
        df = pm.trace_to_dataframe(trace_standard)
        df.to_csv(output_dir + 'data.csv')

        DeltaF.append(df['DeltaF'].median())
        Fb.append(df['Fb'].median())
        t0.append(df['t0'].median())
        u0.append(df['u0'].median())
        tE.append(df['tE'].median())
        K.append(df['K'].median())

    # Save medians of samples
    np.save('output/DeltaF_medians.npy', DeltaF)