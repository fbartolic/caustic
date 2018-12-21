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
import arviz as az
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import exoplanet as xo
from exoplanet.gp import terms, GP
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from Data import OGLEData
from SingleLensModels import PointSourcePointLensMatern32


mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16

events = [] # event names
lightcurves = [] # data for each event

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

data_path = '/home/star/fb90/data/OGLE_ews/2017/'

# Iterate over events, load data, and make plots 
for entry in os.scandir('output'):
    if entry.is_dir():
        data_dir =  data_path + 'blg-' + entry.name[-4:]
        event = OGLEData(data_dir)

        # Plot median models
        #quantiles = np.percentile(samples, [16, 50, 84], axis=0)
        #quantiles_GP = np.percentile(samples_GP, 
            #[16, 50, 84], axis=0)
        t_ = np.linspace(event.df['HJD - 2450000'].values[0], 
                    event.df['HJD - 2450000'].values[-1], 5000)

        ## Calculate residuals
#        model_emcee = PointSourcePointLens_emcee(t, F, sigF)
#        residuals = F - \
#            model_emcee.forward_model(quantiles[1], t)
#
#        fig, ax = plot_data_and_median_model(t, F, sigF, 
#            model_emcee.forward_model(quantiles[1], t_), 
#            residuals, 'NUTS')
#        plt.savefig(entry.path + '/PointSourcePointLens' + '/model.png')    
#
        # Set up GP model
        samples_dir = 'output/' + entry.name + '/PointSourcePointLensGP/' 
        model_matern32 = PointSourcePointLensMatern32(event, parametrization='other')

        with model_matern32:
            trace = pm.load_trace(samples_dir + 'model.trace') 
            print(pm.summary(trace))

        def plot_residuals(ax, pm_model, trace, data):
            df = data.get_standardized_data()

            samples = pm.trace_to_dataframe(trace).values
            print(np.shape(samples))
            quantiles = np.percentile(samples, [16, 50, 84], axis=0)
            median_params = quantiles[1]

            t = df['HJD - 2450000'].values

            with pm_model:
                pred = model_matern32.gp.predict(t, return_var=True)
                Fb = model_matern32.Fb
                u0 = model_matern32.u0
                t0 = model_matern32.t0
                DeltaF = model_matern32.DeltaF
                tE = model_matern32.tE
                u = T.sqrt(u0**2 + ((t - t0)/tE)**2)
                A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
                mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

                pred_mu, pred_var = xo.eval_in_model(pred, median_params)
                mean_function = xo.eval_in_model(mean_func, median_params)

            # Plot the predictions
            mu = mean_function + pred_mu
            residuals = df['I_flux'].values - mu
            ax.plot(t, residuals, 'k.')

        fig, ax = plt.subplots(figsize=(25, 6))
        plot_residuals(ax, model_matern32, trace, event)

        def plot_gp_model(ax, pm_model, trace, t_):
            # Generate 50 realizations of the prediction sampling randomly from the chain
            N_pred = 50
            pred_mu = np.empty((N_pred, len(t_)))
            pred_var = np.empty((N_pred, len(t_)))
            mean_function = np.empty((N_pred, len(t_)))

            with pm_model:
                pred = model_matern32.gp.predict(t_, return_var=True)
                for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                        size=N_pred)):
                    Fb = model_matern32.Fb
                    u0 = model_matern32.u0
                    t0 = model_matern32.t0
                    DeltaF = model_matern32.DeltaF
                    tE = model_matern32.tE
                    u = T.sqrt(u0**2 + ((t_ - t0)/tE)**2)
                    A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
                    mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

                    pred_mu[i], pred_var[i] = xo.eval_in_model(pred, sample)
                    mean_function[i] = xo.eval_in_model(mean_func, sample)

            # Plot the predictions
            for i in range(len(pred_mu)):
                mu = mean_function[i] + pred_mu[i]
                ax.plot(t_, mu, color='C1', alpha=0.2)

        fig, ax = plt.subplots(figsize=(25, 6))
        plot_gp_model(ax, model_matern32, trace, t_)
        event.plot_standardized_data(ax)

        plt.savefig(samples_dir + 'model.png')

        # Plot part of lightcurve
        ti = event.df['HJD - 2450000'].values[0]
        tf = event.df['HJD - 2450000'].values[800]
        t_ = np.linspace(ti, tf, 5000)

        fig, ax = plt.subplots(figsize=(25, 6))
        plot_gp_model(ax, model_matern32, trace, t_)
        event.plot_standardized_data(ax)
        ax.set_xlim(ti, tf)
        plt.savefig(samples_dir + 'model_detail.png')

#        # Plot detail
#        t_ = np.linspace(event.df['HJD - 2450000'].values[0], 
#                    event.df['HJD - 2450000'].values[800], 5000)
#
#        plt.clf()
#        with model_matern32:
#            pred2 = model_matern32.gp.predict(t_, return_var=True)
#            for i, sample in enumerate(xo.get_samples_from_trace(trace, size=N_pred)):
#                Fb = model_matern32.Fb
#                u0 = model_matern32.u0
#                t0 = model_matern32.t0
#                DeltaF = model_matern32.DeltaF
#                tE = model_matern32.tE
#                u = T.sqrt(u0**2 + ((t_ - t0)/tE)**2)
#                A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
#                mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb
#
#                pred_mu[i], pred_var[i] = xo.eval_in_model(pred2, sample)
#                mean_function[i] = xo.eval_in_model(mean_func, sample)
#
#        # Plot the predictions
#        event.plot_standardized_data(ax)
#        fig, ax = plt.subplots(figsize=(25, 6))
#        for i in range(len(pred_mu)):
#            mu = mean_function[i] + pred_mu[i]
#            ax.plot(t_, mu, color='C1', alpha=0.2)
#
#        ax.set_xlim(event.df['HJD - 2450000'].values[0],
#            event.df['HJD - 2450000'].values[800])
#
#        plt.savefig(samples_dir + 'model_detail.png')
#



#        # Plot a Quantile-Quantile plot (QQ plot) for non-GP model
#        plt.clf()
#        fig, ax = plt.subplots(figsize=(6,6))
#        qq_plot(ax, residuals)
#        plt.savefig(entry.path + '/PointSourcePointLens' + '/QQ_plot.png')    
#        
#
#        # Plot samples from posterior in data space for GP model
#        plt.clf()
#        fig, ax = plt.subplots(figsize=(15, 6))
#        plot_data(ax, t, F, sigF) # Plot data
#
#        for s in samples_GP[np.random.randint(len(samples_GP), 
#                size=50)]:
#            gp.set_parameter_vector(np.delete(s, 2))
#            gp.compute(t, s[2]*sigF)
#            mu = gp.predict(F, t_, return_cov=False)
#            ax.plot(t_, mu, color='C1', alpha=0.3)
#            
#        ax.grid(True)
#        plt.savefig(entry.path + '/PointSourcePointLensGP' + '/model_GP.png')    
#
#        # Plot corner plot
#        plt.clf()
#        fig = corner.corner(samples.reshape(-1, len(labels)), 
#            labels=labels)
#        fig.constrained_layout = True
#        plt.savefig(entry.path + '/PointSourcePointLens' + '/corner.png')
#
#        plt.clf()
#        fig = corner.corner(samples_emcee_GP.reshape(-1, len(labels_GP)), 
#            labels=labels_GP)
#        fig.constrained_layout = True
#        plt.savefig(entry.path + '/PointSourcePointLensGP' + '/corner.png')
#
        # Violin plot for important parameters
        ## Seaborn requires that both arrays with posterior samples have the
        ## same dimension
        #n = len(samples_GP[:, -2]) - len(samples[:, -2])
        #print(len(samples[:, -2]))
        #print(len(samples_GP[n:, -2]))
        #df = pd.DataFrame(data=np.stack((samples[:, -2], 
            #samples_GP[n:, -2]), axis=1), columns=["no GP", "GP"])
        
        #plt.clf()
        #fig, ax = plt.subplots(figsize=(8, 6))
        #ax = sns.violinplot(data=df)
        #ax.set_xlabel('Model')
        #ax.set_ylabel(r'$t_E$ [days]')
        #ax.grid(True)

        #plt.savefig(entry.path  + '/tE_posterior.png')