import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import os
import sys
sys.path.append("../../exoplanet")
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
from SingleLensModels import PointSourcePointLensMatern32
from SingleLensModels import PointSourcePointLensSHO
from SingleLensModels import PointSourcePointLensSHOProduct
from SingleLensModels import PointSourcePointLensStudentT

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

def evaluate_gp_model_on_grid(data, pm_model, trace, t_grid, N_pred):
    # Generate 50 realizations of the prediction sampling randomly from the chain
    gp_pred_mu = np.empty((N_pred, len(t_grid)))
    gp_pred_var = np.empty((N_pred, len(t_grid)))
    mean_function = np.empty((N_pred, len(t_grid)))

    with pm_model:
        # Evaluate model
        gp_pred = pm_model.gp.predict(t_grid, return_var=True)
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

            gp_pred_mu[i], gp_pred_var[i] = xo.eval_in_model(gp_pred,
                sample)
            mean_function[i] = xo.eval_in_model(mean_func, sample)

    return mean_function + gp_pred_mu
        
def evaluate_median_gp_model_on_grid(data, pm_model, trace, t_grid):
    median_params_dict = {}
    for i in range(len(pm_model.vars)):
        median_params_dict[str(pm_model.vars[i])] = \
            np.percentile(trace[pm_model.vars[i]], [16, 50, 84], 
            axis=0)[1]

    with pm_model:
        # Evaluate residuals
        gp_pred = pm_model.gp.predict(t_grid, return_var=True)
        Fb = pm_model.Fb
        u0 = pm_model.u0
        t0 = pm_model.t0
        DeltaF = pm_model.DeltaF
        tE = pm_model.tE
        u = T.sqrt(u0**2 + ((t_grid - t0)/tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
        mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

        gp_pred_mu, gp_pred_var = xo.eval_in_model(gp_pred, 
            median_params_dict)
        mean_function = xo.eval_in_model(mean_func, 
            median_params_dict)

    return mean_function + gp_pred_mu

def plot_gp_model_and_residuals(ax, event, pm_model, trace, ti_idx,
    tf_idx):
    N_pred = 50

    df = event.get_standardized_data()
    time = df['HJD - 2450000'].values

    t_grid = np.linspace(time[ti_idx], time[tf_idx], 1000)
    t = time[ti_idx:tf_idx]

    predictions = evaluate_gp_model_on_grid(event, pm_model, trace,
        t_grid, N_pred)
    
    mask = ((time >= time[ti_idx]) & (time <= time[tf_idx]))
    event.plot_standardized_data(ax[0], mask)
    ax[0].set_xlabel(None)

    for i in range(N_pred):
        ax[0].plot(t_grid, predictions[i, :], color='C1', alpha=0.2)

    # Plot residuals
    df = event.get_standardized_data()
    residuals = df['I_flux'].values[ti_idx:tf_idx] -\
        evaluate_median_gp_model_on_grid(event, pm_model, trace, t)
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

# Iterate over events, load data, and make plots 
for entry in os.scandir('output'):
    if entry.is_dir():
        data_dir = data_path + 'blg-' + entry.name[-4:]
        event = OGLEData(data_dir)
        print(event.event_name)

        # Load traces
        output_dir = 'output/' + entry.name + '/PointSourcePointLens/' 
#        output_dir_matern32 = 'output/' + entry.name + '/PointSourcePointLensGP/' 
#        output_dir_SHO = 'output/' + entry.name + '/PointSourcePointLensSHO/' 
#        output_dir_SHO_product = 'output/' + entry.name +\
#            '/PointSourcePointLensSHOProduct/' 
        output_dir_studentT = 'output/' + entry.name + '/PointSourcePointLensStudenT'

        model_standard = PointSourcePointLens(event, use_joint_prior=True)
#        model_matern32 = PointSourcePointLensMatern32(event, 
#            use_joint_prior=True)
#        model_SHO = PointSourcePointLensSHO(event, use_joint_prior=True)
#        model_SHO_product = PointSourcePointLensSHOProduct(event, 
#            use_joint_prior=True)
        model_standard_studentT = PointSourcePointLensStudentT(event, use_joint_prior=True)

        with model_standard:
            trace_standard = pm.load_trace(output_dir + 'model.trace') 
#        
#        with model_matern32:
#            trace_matern32 = pm.load_trace(output_dir_matern32 + 'model.trace') 
#       
#        with model_SHO:
#            trace_SHO = pm.load_trace(output_dir_SHO + 'model.trace') 
#        
#        with model_SHO_product:
#            trace_SHO_product = pm.load_trace(output_dir_SHO_product + 'model.trace') 
        
        with model_standard_studentT:
            trace_standard_studentT = pm.load_trace(output_dir_studentT + 'model.trace') 

        # Plot traceplots
#        plot_traceplots(trace_standard, 7, output_dir)
#        plot_traceplots(trace_matern32, 11, output_dir_matern32)
#        plot_traceplots(trace_SHO, 10, output_dir_SHO)
#        plot_traceplots(trace_SHO_product, 10, output_dir_SHO_product)
        plot_traceplots(trace_standard_studentT, 8, output_dir_studentT)

        # Plot non-GP models
#        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#             figsize=(25, 10), sharex=True)
#        fig.subplots_adjust(hspace=0.05)
#        plot_model_and_residuals(ax, event, model_standard, trace_standard, 0,
#            len(event.df['HJD - 2450000']) - 1)
#        plt.savefig(output_dir + 'model.pdf')

        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
             figsize=(25, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)
        plot_model_and_residuals(ax, event, model_standard_studentT, trace_standard_studentT, 0,
            len(event.df['HJD - 2450000']) - 1)
        plt.savefig(output_dir + 'model.pdf')

        # QQ plot 
#        df = event.get_standardized_data()
#        residuals = df['I_flux'].values -\
#            evaluate_median_model_on_grid(event, model_standard, 
#            trace_standard, df['HJD - 2450000'].values)
#        fig, ax = plt.subplots(figsize=(6,6))
#        qq_plot(ax, residuals)
#        plt.savefig(entry.path + '/PointSourcePointLens' + '/QQ_plot.png')    

        # Plot GP models
        # Plot GP model for complete lightcurve
#        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#             figsize=(25, 10), sharex=True)
#        fig.subplots_adjust(hspace=0.05)
#        plot_gp_model_and_residuals(ax, event, model_matern32, trace_matern32, 0,
#            len(event.df['HJD - 2450000']) - 1)
#        plt.savefig(output_dir_matern32 + 'model.pdf')
#
#        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#             figsize=(25, 10), sharex=True)
#        fig.subplots_adjust(hspace=0.05)
#        plot_gp_model_and_residuals(ax, event, model_SHO, trace_SHO, 0,
#            len(event.df['HJD - 2450000']) - 1)
#        plt.savefig(output_dir_SHO + 'model.pdf')
#
#        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#             figsize=(25, 10), sharex=True)
#        fig.subplots_adjust(hspace=0.05)
#        plot_gp_model_and_residuals(ax, event, model_SHO_product, 
#            trace_SHO_product, 0, len(event.df['HJD - 2450000']) - 1)
#        plt.savefig(output_dir_SHO_product + 'model.pdf')
#
#        # Plot model for part of the lightcurve 
#        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#             figsize=(25, 10), sharex=True)
#        fig.subplots_adjust(hspace=0.05)
#        plot_gp_model_and_residuals(ax, event, model_matern32, trace_matern32, 0, 200)
#        plt.savefig(output_dir_matern32 + 'model_detail.pdf')
#
#        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#             figsize=(25, 10), sharex=True)
#        fig.subplots_adjust(hspace=0.05)
#        plot_gp_model_and_residuals(ax, event, model_SHO, trace_SHO, 0, 200)
#        plt.savefig(output_dir_SHO + 'model_detail.pdf')
#
#        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
#             figsize=(25, 10), sharex=True)
#        fig.subplots_adjust(hspace=0.05)
#        plot_gp_model_and_residuals(ax, event, model_SHO_product,
#            trace_SHO_product, 0, 200)
#        plt.savefig(output_dir_SHO_product + 'model_detail.pdf')
#
#
        # Violin plot for important parameters
        # Seaborn requires that both arrays with posterior samples have the
        # same dimension
        tE_samples = trace_standard['tE']
        #tE_samples_gp = trace_matern32['tE']
        #tE_samples_SHO = trace_SHO['tE']
        #tE_samples_SHO_product = trace_SHO_product['tE']
        tE_samples_studentT = trace_standard_studentT['tE']

        #df = pd.DataFrame(data=np.stack((tE_samples, tE_samples_gp[:],
        #    tE_samples_SHO, tE_samples_SHO_product), axis=1), 
        #    columns=["no GP", "Matern32", "SHO", "SHO Product"])

        df = pd.DataFrame(data=np.stack((tE_samples,
            tE_samples_studentT), axis=1), 
            columns=["no GP", 'rescaled'])

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.violinplot(data=df)
        ax.set_xlabel('Model')
        ax.set_ylabel(r'$t_E$ [days]')
        ax.grid(True)
#        sigtE = np.std(tE_samples_gp)
#        median_tE = np.median(tE_samples_gp)
#        ax.set_ylim(0, median_tE + 5*sigtE)

        plt.savefig('output/' + entry.name + '/tE_posteriors.pdf')