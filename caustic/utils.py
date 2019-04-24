import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import exoplanet as xo
import pymc3 as pm
import theano.tensor as T
import os

from data import OGLEData

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16

def qq_plot(ax, residuals):
    """Plots quantile-quantile plot of residuals vs draws from a Gaussian."""
    percentiles = np.linspace(0, 100, 500)
    quantile_residuals = np.percentile(residuals, percentiles)
    quantile_model = np.percentile(np.random.normal(size=10000), percentiles)

    ax.scatter(quantile_model, quantile_residuals, marker='o', color='black',
        alpha=0.3)
    ax.grid()
    x = np.linspace(quantile_model[0], quantile_model[-1])
    # ax.plot(x, x, color='black', linestyle='dashed', alpha=0.8)
    ax.set_aspect(1)
    ax.set_xlabel("Modeled residuals")
    ax.set_ylabel("Measured residuals")

def plot_model_and_residuals(ax, event, pm_model, trace_path, n_samples):
    """
    Plots model in data space given samples from the posterior 
    distribution. Also plots residuals with respect to the median model, where 
    the median model is the median of multiple posterior draws of the model 
    in data space, rather then a single draw corresponding to median values of
    all parameters.
    
    Parameters
    ----------
    ax : matplotlib.axes 
        Needs to be of shape (2, 1).
    event : caustic.data 
        Microlensing event data. 
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    trace_path : str
        Path to PyMC3 trace object saved as 'model.trace'.
    n_samples: int
        Number of posterior draws to be plotted.
    """
    # Load standardized data
    tables = event.get_standardized_data()
    t_grids = [T._shared(np.linspace(table['HJD'][0], table['HJD'][-1], 2000))\
        for table in tables]
    t_observed = [T._shared(table['HJD']) for table in tables]

    # Compute model predictions on a fine grid and at observed times
    with pm_model as model_instance:
        trace = pm.load_trace(trace_path) 
        pred = model_instance.evaluate_model_on_grid(trace, t_grids, 
            n_samples=50)
        pred_at_observed_times = model_instance.evaluate_model_on_grid(trace,
            t_observed, n_samples=100) 

    # Plot data
    event.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')

    n_bands = len(tables)

    # Plot predictions for various samples
    for n in range(n_bands): # iterate over bands
        for i in range(len(pred[n, :, 0])):
            ax[0].plot(t_grid, pred[n, i, :], color='C' + str(n), 
                alpha=0.2)

    # Calculate and plot residuals
    for n in range(n_bands): # iterate over bands
        quantile_predictions = np.percentile(pred_at_observed_times[n],
            [16, 50, 84], axis=1)

        residuals =  tables[n]['flux'] - quantile_predictions[1, n, :]
        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='.', color='C' + str(n))
        ax[1].grid(True)

def plot_map_model_and_residuals(ax, event, pm_model, map_point):
    """
    Plots model in data space given MAP parameters of the model. Also plots 
    residuals with respect to MAP model.
    
    Parameters
    ----------
    ax : matplotlib.axes 
        Needs to be of shape (2, 1).
    event : caustic.data 
        Microlensing event data. 
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    map_point : dict 
        Dictionary containing the MAP values of model parameters.
    """
    # Load standardized data
    tables = event.get_standardized_data()
    t_grid = np.linspace(tables[0]['HJD'][0], 
            tables[0]['HJD'][-1], 2000)

    # Compute model predictions on a fine grid and at observed times
    with pm_model as model_instance:
        pred = model_instance.evaluate_map_model_on_grid(t_grid, map_point)
        pred_at_observed_times = [model_instance.evaluate_map_model_on_grid(
            table['HJD'], map_point) for table in tables]

    # Plot data
    event.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')

    n_bands = len(tables)

    # Plot predictions for various samples
    for n in range(n_bands): # iterate over bands
        ax[0].plot(t_grid, pred[n, :], color='C' + str(n), alpha=0.5)
        print(pred[n, :])

    # Calculate and plot residuals
    for n in range(n_bands): # iterate over bands
        residuals =  tables[n]['flux'] - pred_at_observed_times[n][n]
        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='.', color='C' + str(n))
        ax[1].grid(True)

def plot_prior_model_samples(ax, event, pm_model, n_samples):
    """
    Plots model in data space given samples drawn from the prior probability
    distribution. This is useful debugging prior choices.
    
    Parameters
    ----------
    ax : matplotlib.axes
        Single matplotlib axes object.
    event : caustic.data
        Microlensing event data. Although this functions plots draws from the
        prior, some of the priors depend on the data in a simple way so a
        data object is still needed.
    pm_model : pymc3.Model
        PyMC3 model object. Needs to work with `sample_prior_predictive` method
        from PyMC3.
    n_samples : int
        Number of samples from prior.
    """
    # Load standardized data
    tables = event.get_standardized_data()
    t_grids = [T._shared(np.linspace(table['HJD'][0], table['HJD'][-1], 2000))\
        for table in tables]

    # Sample from the prior
    with pm_model as model_instance:
        trace = pm.sample_prior_predictive(n_samples)
        predictions = model_instance.evaluate_prior_model_on_grid(trace, t_grids,
             n_samples)

    # Plot data
    event.plot_standardized_data(ax)
    ax.set_xlabel('HJD - 2450000')

    # Plot predictions for various samples
    for i in range(n_samples):
        for n in range(model_instance.n_bands): # iterate over bands
            ax.plot(t_grids[n].eval(), predictions[n][i, :], color='C' + str(n), 
                alpha=0.5)

def plot_histograms_of_prior_samples(event, pm_model, output_dir):
    """
    Plots histograms of draws from prior distribution for each model parameter.
    The plots are saved to disk.
    
    Parameters
    ----------
    event : caustic.data
        Microlensing event data. Although this functions plots draws from the
        prior, some of the priors depend on the data in a simple way so a
        data object is still needed.
    pm_model : pymc3.Model
        PyMC3 model object. Needs to work with `sample_prior_predictive` method
        from PyMC3.
    output_dir : str 
        Path to output directory where the images are going to be saved.
    """
    # Sample from the prior
    n_samples = 5000
    with pm_model as model_instance:
        trace = pm.sample_prior_predictive(n_samples)

    directory = output_dir + '/prior_samples/'
    # Create output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key in trace.keys():
        value = trace[key]
        # Some parameters are multivariate and in those cases the keys have 
        # different dimensions
        if value.ndim > 1:
            for i in range(value.ndim):
                fig, ax = plt.subplots() 
                label = key + str(i)
                ax.hist(value[:, i], bins=30);
                ax.set_xlabel(label)
                plt.savefig(directory + label + '.png')
        else:
            fig, ax = plt.subplots() 
            label = key
            ax.hist(value, bins=30);
            plt.savefig(directory + label + '.png')

def plot_violin_plots(ax, pm_models, trace_paths, parameter_names):
    for idx, model in enumerate(pm_models):
        with pm_model as model_instance:
            trace = pm.load_trace(trace_path) 

    return 0