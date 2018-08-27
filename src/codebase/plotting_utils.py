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

def plot_emcee_traceplots(sampler, acceptance_fraction=0, labels):
    """Plots MCMC traceplot given an emcee sampler object.
    
    Parameters
    ----------
    sampler : object
        emcee sampler object
    acceptance_fraction : float
        Plot only the chains which have an avg. acceptance rate above a certain
        treshold, defaults to zero.
    labels : array
        Array of latex strings for parameter labels.
    
    """

    n_pars = len(sampler.flatchain[0, :])
    if len(labels)!=n_pars:
        raise ValueError("Number of labels has to be equal to the number of \
            parameters.")

    fig, ax = plt.subplots(n_pars, sharex=True, figsize=(15,10))
    
    mask = sampler.acceptance_fraction > acceptance_fraction

    fig.subplots_adjust(hspace=0.08)

    for i in range(n_pars):
        tmp = sampler.chain[mask, ::1, i].T
        ax[i].plot(np.arange(len(tmp)), tmp, 'k-', alpha=0.2);
        ax[i].set_ylabel(labels[i])    

    for a in ax.ravel():
        a.grid(True)

    ax[-1].set_xlabel('steps')

