import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import exoplanet as xo
import pymc3 as pm
import theano.tensor as T
import os

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 18

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

def plot_model_and_residuals(ax, event, pm_model, trace_path, n_samples=50,
        **kwargs):
    """
    Plots model in data space given samples from the posterior 
    distribution. Also plots residuals with respect to the median model, where 
    the median model is the median of multiple posterior draws of the model 
    in data space, rather then a single draw corresponding to median values of
    all parameters. All extra keyword arguments are passed to the matplotlib
    plot function.
    
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

    # Compute model predictions on a fine grid and at observed times
    with pm_model as model_instance:
        t_grids = [np.linspace(model_instance.t_begin, model_instance.t_end,
            5000) for table in tables]
        t_observed = [np.array(table['HJD']) for table in tables]

        trace = pm.load_trace(trace_path) 
        pred = model_instance.evaluate_posterior_model_on_grid(trace, t_grids, 
            n_samples)
        pred_at_observed_times = model_instance.evaluate_posterior_model_on_grid(trace,
            t_observed, n_samples=100) # more samples needed for accurate median

    # Plot data
    event.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')

    for a in ax.ravel():
        a.set_xlim(t_grids[0][0], t_grids[0][-1])
#        a.set_xlim(7820, 8020)

    # Plot predictions for various samples
    for n in range(model_instance.n_bands): # iterate over bands
        for i in range(n_samples):
            ax[0].plot(t_grids[n], pred[n][i, :], color='C' + str(n), 
                alpha=0.2)
    # Calculate and plot residuals
    for n in range(model_instance.n_bands): # iterate over bands
        quantile_predictions = np.percentile(pred_at_observed_times[n],
            [16, 50, 84], axis=0)

        residuals =  tables[n]['flux'] - quantile_predictions[1]
        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='o', color='C' + str(n), alpha=0.5)
        ax[1].grid(True)

def plot_map_model_and_residuals(ax, event, pm_model, map_point, **kwargs):
    """
    Plots model in data space given MAP parameters of the model. Also plots 
    residuals with respect to MAP model. All extra keyword arguments are passed
    to the matplotlib plot function.
    
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
    
    # Compute model predictions on a fine grid and at observed times
    with pm_model as model_instance:
        t_grids = [np.linspace(model_instance.t_begin, model_instance.t_end,
            5000) for table in tables]
        t_observed = [table['HJD'] for table in tables]

        pred = model_instance.evaluate_map_model_on_grid(t_grids, map_point)
        pred_at_observed_times = model_instance.evaluate_map_model_on_grid(
            t_observed, map_point)

    # Plot data
    event.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')

    # Plot predictions for various samples
    for n in range(model_instance.n_bands): # iterate over bands
        ax[0].plot(t_grids[n], pred[n], color='C' + str(n), **kwargs)

    # Calculate and plot residuals
    for n in range(model_instance.n_bands): # iterate over bands
        residuals = np.array(tables[n]['flux']) - pred_at_observed_times[n]
        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='o', color='C' + str(n), alpha=0.5, **kwargs)
        ax[1].grid(True)

def plot_prior_model_samples(ax, event, pm_model, n_samples, **kwargs):
    """
    Plots model in data space given samples drawn from the prior probability
    distribution. This is useful debugging prior choices. All extra keyword
    arguments are passed to the matplotlib plot function.
    
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
    t_grids = [np.linspace(table['HJD'][0], table['HJD'][-1], 2000)\
        for table in tables]

    # Sample from the prior
    with pm_model as model_instance:
        trace = pm.sample_prior_predictive(n_samples)
        predictions = model_instance.evaluate_prior_model_on_grid(trace, t_grids)

    # Plot predictions for various samples
    for i in range(n_samples):
        for n in range(model_instance.n_bands): # iterate over bands
            ax.plot(t_grids[n], predictions[n][i, :], color='C' + str(n), 
                alpha=0.5, **kwargs)

    ax.set_xlabel('HJD - 2450000')

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
    with pm_model:
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
            for i in range(len(value[0])):
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

def plot_trajectory(ax, pm_model, trace_path, n_samples=50, **kwargs):
    """
    Plots different trajectories of the lens w.r. top the source on the plane
    of the sky in (u_n, u_e) coordinates. All extra keyword arguments are 
    passed to the plotting function.

    Parameters
    ----------
    ax : matplotlib.axes 
        Needs to be of shape (1, 1).
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    trace_path : str
        Path to PyMC3 trace object saved as 'model.trace'.
    n_samples: int
        Number of posterior draws to be plotted.
    """
    # Compute model predictions on a fine grid 
    with pm_model as model_instance:
        t_grid = np.linspace(model_instance.t_begin, model_instance.t_end, 2000)
        trace = pm.load_trace(trace_path) 
        pred = model_instance.evaluate_trajectory_on_grid(trace, t_grid, 
            n_samples)

    ax.set_xlabel(r'$u_e\,[\theta_E]$')
    ax.set_ylabel(r'$u_n\,[\theta_E]$')
    ax.axhline(0., color='grey')
    ax.axvline(0., color='grey')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Source star
    circle = plt.Circle((0., 0.), 0.1, color='C1', zorder=2)
    ax.add_artist(circle)

    ax.grid()

    # Plot predictions for various samples
    for i in range(n_samples):
        ax.plot(pred[i, 0, :], pred[i, 1, :], alpha=0.2, **kwargs)

def remove_outliers(model, event):
    """
    Optimizes a flexible PSPL model including a Gaussian Process in order to 
    flag any outliers.
    
    Parameters
    ----------
    model : pymc3.Model 
        PyMC3 Model object.
    event : caustic.data 
        Microlensing event data. 
    """
    # Remove worst outliers using rolling MAD before doing optimization
    event.remove_worst_outliers()

    # Optimize a flexible GP model model
    with model:
        start = model.test_point
        map_soln = xo.optimize(start=start, vars=[model.F_base])
        map_soln = xo.optimize(start=map_soln, vars=[model.Delta_F])
        map_soln = xo.optimize(start=map_soln, vars=[model.u0])
        map_soln = xo.optimize(start=map_soln, vars=[model.t0])
        map_soln = xo.optimize(start=map_soln, vars=[model.u0, model.teff])
        map_soln = xo.optimize(start=map_soln)

        t_observed = [T._shared(table['HJD']) for table in event.tables]
        pred_at_observed_times = model.evaluate_map_model_on_grid(
            t_observed, map_soln)

    # Reomove all points deviating from MAP model by x*sigma_MAD
    for n in range(model.n_bands):
        resid = event.tables[n]['flux'] - pred_at_observed_times[n]
        mad = lambda x: 1.4826*np.median(np.abs(x - np.median(x)))
        mask = np.abs(resid/mad(resid)) < 7

        # Updata mask
        event.masks[n] = mask

    # Plot data without outliers
    fig, ax = plt.subplots(figsize=(25, 10))
    event.plot(ax)

#def plot_violin_plots(ax, pm_models, trace_paths, parameter_names):
#    for idx, model in enumerate(pm_models):
#        with pm_model as model_instance:
#            trace = pm.load_trace(trace_path) 
#
#    return 0