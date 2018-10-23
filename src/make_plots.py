import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy
import corner
import emcee
import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import seaborn as sns
import os
import sys
sys.path.append('codebase')
from data_preprocessing_ogle import process_data
from plotting_utils import *

from emcee_model import PointSourcePointLens_emcee
from emcee_gp_model import PointSourcePointLensGP_emcee

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

# Iterate over events, load data, and make plots 
for entry in os.scandir('output'):
    if entry.is_dir():
        data = np.load(entry.path + '/data.npy')
        t, F, sigF = data[:, 0], data[:, 1], data[:, 2]
        
        # Load emcee model
        model_emcee = PointSourcePointLens_emcee(t, F, sigF)

        # Load emcee model with GP
        model_emcee_GP = PointSourcePointLensGP_emcee(t, F, sigF)

        # Load posterior samples
        samples_pymc3 = np.load(entry.path + '/samples_pymc3.npy')
        samples_emcee = np.load(entry.path + '/samples_emcee.npy')
        samples_emcee_GP = np.load(entry.path + '/samples_emcee_GP.npy')
        samples_emcee_acc_frac = np.load(entry.path + \
            '/samples_emcee_acc_frac.npy')
        samples_emcee_GP_acc_frac = np.load(entry.path + \
            '/samples_emcee_GP_acc_frac.npy')
        
        # Plot traceplots
        labels = ['$\Delta F$', '$F_b$', '$t_0$', '$t_{eff}$', '$t_E$', '$u_K$']
        labels_GP = ['$\ln\sigma$', 'ln_rho','$\Delta F$', 
            '$F_b$', '$t_0$', '$t_{eff}$', '$t_E$', '$u_K$']

        fig1, ax1 = plot_emcee_traceplots(samples_emcee,
            labels, samples_emcee_acc_frac, acceptance_fraction_cutoff=0.05)
        plt.savefig(entry.path + '/emcee_traceplots.png')    

        fig2, ax2 = plot_emcee_traceplots(samples_emcee_GP,
            labels_GP, samples_emcee_GP_acc_frac, 
                acceptance_fraction_cutoff=0.05)
        plt.savefig(entry.path + '/emcee_GP_traceplots.png')    

        # Plot median models
        quantiles_pymc3 = np.percentile(samples_pymc3, [16, 50, 84], axis=0)
        quantiles_emcee = np.percentile(samples_emcee.reshape(-1, len(labels)), 
            [16, 50, 84], axis=0)
        quantiles_emcee_GP = \
            np.percentile(samples_emcee_GP.reshape(-1, len(labels_GP)), 
            [16, 50, 84], axis=0)

        t_ = np.linspace(t[0], t[-1], 1000)

        plt.clf()

        ## Calculate residuals
        residuals_pymc3 = F - \
            model_emcee.forward_model(quantiles_pymc3[1], t)
        residuals_emcee = F - \
            model_emcee.forward_model(quantiles_emcee[1], t)

        fig, ax = plot_data_and_median_model(t, F, sigF, 
            model_emcee.forward_model(quantiles_pymc3[1], t_), 
            residuals_pymc3, 'NUTS')
        plt.savefig(entry.path +  '/model_pymc3.png')    

        fig, ax = plot_data_and_median_model(t, F, sigF, 
            model_emcee.forward_model(quantiles_emcee[1], t_), 
            residuals_emcee, 'emcee')
        plt.savefig(entry.path +  '/model_emcee.png')    

        ## Calculate median GP model
        samples_GP = samples_emcee_GP.reshape(-1, len(labels_GP))
        model_emcee_GP = PointSourcePointLensGP_emcee(t, F, sigF)
        quantiles_emcee_GP = np.percentile(samples_GP,
             [16, 50, 84], axis=0)

        median_GP_params = quantiles_emcee_GP[1, :]
        gp = model_emcee_GP.gp
        gp.set_parameter_vector(median_GP_params[:-1])
        u_K = median_GP_params[-1]
        if u_K < 0:
            K = 1.
        else:
            K = 1 - np.log(1 - u_K)

        gp.compute(t, K*sigF)
        mu = gp.predict(F, t_, return_cov=False)
        residuals_GP = F - gp.predict(F, t, return_cov=False)

        plt.clf()
        fig, ax = plot_data_and_median_model(t, F, sigF, mu,
            residuals_GP, 'GP model')
        plt.savefig(entry.path +  '/model_GP_median.png')    
        
        # Plot a Quantile-Quantile plot (QQ plot)
        plt.clf()
        fig, ax = plt.subplots(figsize=(6,6))
        qq_plot(ax, residuals_pymc3)
        plt.savefig(entry.path + '/QQ_plot.png')    
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(15, 6))
        plot_data(ax, t, F, sigF) # Plot data

        for s in samples_GP[np.random.randint(len(samples_GP), 
                size=50)]:
            gp.set_parameter_vector(s[:-1])
            gp.compute(t, s[-1]*sigF)
            mu = gp.predict(F, t_, return_cov=False)
            ax.plot(t_, mu, color='C1', alpha=0.3)
            
        ax.grid(True)
        plt.savefig(entry.path + '/model_GP.png')    

        # Plot corner plot
        plt.clf()
        fig = corner.corner(samples_pymc3.reshape(-1, len(labels)), 
            labels=labels)
        fig.constrained_layout = True
        plt.savefig(entry.path + '/corner_pymc3.png')

        plt.clf()
        fig = corner.corner(samples_emcee.reshape(-1, len(labels)),
             labels=labels)
        fig.constrained_layout = True
        plt.savefig(entry.path + '/corner_emcee.png')

        plt.clf()
        fig = corner.corner(samples_emcee_GP.reshape(-1, len(labels_GP)), 
            labels=labels_GP)
        fig.constrained_layout = True
        plt.savefig(entry.path + '/corner_emcee_GP.png')

        # Violin plot for important parameters
        ## Seaborn requires that both arrays with posterior samples have the
        ## same dimension
        n = len(samples_GP[:, -2]) - len(samples_pymc3[:, -2])
        print(len(samples_pymc3[:, -2]))
        print(len(samples_GP[n:, -2]))
        df = pd.DataFrame(data=np.stack((samples_pymc3[:, -2], 
            samples_GP[n:, -2]), axis=1), columns=["no GP", "GP"])
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.violinplot(data=df)
        ax.set_xlabel('Model')
        ax.set_ylabel(r'$t_E$ [days]')
        ax.grid(True)

        plt.savefig(entry.path  + '/tE_posterior.png')