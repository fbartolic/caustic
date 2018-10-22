import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 22

def plot_data(ax, x, y, y_err):
    """Plots time series data with errorbars."""
    ax.errorbar(x, y, y_err, fmt='.', color='black', label='Data', 
        ecolor='#686868')
    ax.grid(True)
    ax.set_xlabel('HJD - 2450000')
    ax.set_ylabel('Normalized flux')

def plot_data_and_median_model(x, y, y_err, model, residuals, 
        label):
    """Plots time series data with errorbars plus a median/mean model."""

    fig, ax = plt.subplots(2, 1, figsize=(20, 8), sharex=True, 
        gridspec_kw = {'height_ratios':[3, 1]})
    fig.subplots_adjust(hspace=0.)

    ax[0].errorbar(x, y, y_err, fmt='.', color='black', label='Data', 
        ecolor='#686868')
    x_ = np.linspace(x[0], x[-1], len(model))
    ax[0].plot(x_, model, marker='', linestyle='-', color='C0', lw=2.)
    ax[0].set_ylabel('Normalized flux')

    ax[1].errorbar(x, residuals, y_err, fmt='.', color='black', 
            ecolor='#686868')

    ax[1].set_ylabel('Residuals')
    
    for a in ax.ravel():
        a.grid(True)
        a.set_xlabel('HJD - 2450000')

    return fig, ax

def plot_emcee_traceplots(samples, labels, acceptance_fractions, 
        acceptance_fraction_cutoff=0):
    """Plots MCMC traceplot given an emcee sampler object.
    
    Parameters
    ----------
    ax : object
        Matplotlib axes object.
    chains : ndarray 
        Chains for all the walkers, shape of the array is (nwalkers, nsamples,
        npars)
    acceptance_fractions : ndarray
        Array of acceptance fractions for each walker.
    acceptance_fraction_cutoff : float
        Plot only the chains which have an avg. acceptance rate above a certain
        treshold, defaults to zero.
    labels : array
        Array of latex strings for parameter labels.
    """

    n_pars = len(samples[0, 0, :])
    if len(labels)!=n_pars:
        raise ValueError("Number of labels has to be equal to the number of \
            parameters.")

    fig, ax = plt.subplots(n_pars, sharex=True, figsize=(15,10))
    
    mask = acceptance_fractions > acceptance_fraction_cutoff

    fig.subplots_adjust(hspace=0.08)

    for i in range(n_pars):
        tmp = samples[mask, :, i].T
        ax[i].plot(np.arange(len(tmp)), tmp, 'k-', alpha=0.2);
        ax[i].set_ylabel(labels[i])    

    for a in ax.ravel():
        a.grid(True)

    ax[-1].set_xlabel('steps')

    return fig, ax

