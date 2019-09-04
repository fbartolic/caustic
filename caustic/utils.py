import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import exoplanet as xo
import pymc3 as pm
import theano.tensor as T
import caustic as ca
import os

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 18

def construct_masked_tensor(array_list):
    """
    Given a list of 1D numpy arrays, this function returns a theano tensor
    of shape (n_elements, n_max) where n_elements is the number of arrays
    in the list, and n_max is the length of the largest array in the list.
    The missing values are filled in with zeros and a mask is returned 
    together with tensor. The purpose of this function is to construct 
    tensors instead of using lists in order to avoid having to do loops.

    Parameters
    ----------
    array_list : list 
        List of numpy arrays of varying lengths. 
    
    Returns
    -------
    tuple 
        Returns tuple (tensor, mask) where tensor and maks are 
        theano.tensor objects of the same shape (n_elements, n_max). 
        The mask tensor is of datta type int8 and the elements are 
        equal to 1 for non-filled in values and zero otherwise. To use
        the mask in theano, use `tensor[mask.nonzero()]`.

    """
    for array in array_list:
        if(array.ndim != 1):
            raise ValueError('Make sure that all of the arrays are 1D and\
            have the same dimension')

    n_max =  np.max([len(array) for array in array_list])
    tensor = T.as_tensor_variable(np.stack([np.pad(array, 
        (0, n_max - len(array)), 'constant', 
        constant_values=(0.,)) for array in array_list]))
    
    masks_list = []

    for array in array_list:
        array = np.append(
            np.ones(len(array)),  
            np.zeros(n_max - len(array))
            )
        masks_list.append(array)

    mask = T.as_tensor_variable(np.stack(masks_list).astype('int8'))

    return tensor, mask

def estimate_t0(event):
    """
    Estimates the initial value for the t0 parameter. This is necessary because
    the posterior is highly multi-modal in t0 and the sampler usually
    fails to converge if t0 is not close to true value.
    """
    event.remove_worst_outliers(window_size=20, mad_cutoff=2)
    tables = event.get_standardized_data()
    fluxes = np.concatenate([table['flux'] for table in tables])
    times = np.concatenate([table['HJD'] for table in tables])

    guess = np.median(times[fluxes > 4])

    cut = 4
    while np.isnan(guess):
        guess = np.median(times[fluxes > cut])
        cut -= 0.5

    return guess

def revert_flux_params_to_nonstandardized_format(data, Delta_F, F_base, u_0):
    # Revert F_base and Delta_F to non-standardized units
    data.units = 'fluxes'
    fluxes_median = np.zeros(len(data.light_curves))
    fluxes_std = np.zeros(len(data.light_curves))

    for i, table in enumerate(data.light_curves):
        mask = table['mask']
        fluxes_median[i] = np.median(table['flux'][mask])
        fluxes_std[i] = np.std(table['flux'][mask])

    # Flux parameters to standard flux units 
    Delta_F_ = T.as_tensor_variable(fluxes_std)*Delta_F 
    F_base_ = T.as_tensor_variable(fluxes_std)*F_base +\
        T.as_tensor_variable(fluxes_median)

    # Calculate source flux and blend flux
    A_u0 = (u_0**2 + 2)/(T.abs_(u_0)*T.sqrt(u_0**2 + 4))

    F_S = Delta_F_/(A_u0 - 1)
    F_B = F_base_ - F_S

    g = F_B/F_S

    # Convert fluxes to magnitudes
    zero_point = 22.
    m_source = zero_point - 2.5*T.log10(F_S)

    return m_source, g
    
def plot_model_and_residuals(ax, data, pm_model, trace, t_grid, prediction, 
        n_samples=50, **kwargs):
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
    data : caustic.data 
        Microlensing event data. 
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    trace : PyMC3 MultiTrace object
        Trace object containing samples from posterior.
    t_grid : theano.tensor
        Times at which we want to evaluate model predictions. Shape 
        (n_bands, n_pts).
    prediction : theano.tensor
        Model prediction evaluated at t_grid.
    n_samples: int
        Number of posterior draws to be plotted.
    """
    # Load standardized data
    tables = data.get_standardized_data()

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()
    n_bands = len(data.light_curves)

    prediction_eval = np.zeros((n_samples, n_bands, n_pts_dense))

    with pm_model:
        for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                size=n_samples)):
            prediction_eval[i] = xo.eval_in_model(prediction, sample)

    # Plot model predictions for each different samples from posterior on dense 
    # grid 
    for n in range(n_bands): # iterate over bands
        for i in range(n_samples):
            ax[0].plot(t_grid[n].eval(), prediction_eval[i, n, :],
                color='C' + str(n), alpha=0.2, **kwargs)

    # Compute median of the predictions
    median_predictions = np.zeros((n_bands, n_pts_dense))
    for n in range(n_bands): 
        median_predictions[n] = np.percentile(prediction_eval[n],
            [16, 50, 84], axis=0)[1]

    # Plot data
    data.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')
                
    # Compute residuals with respect to median model 
    for n in range(n_bands): 
        # Interpolate median predictions onto a grid of observed times
        median_pred_interp = np.interp(tables[n]['HJD'], 
            t_grid[n].eval(), median_predictions[n]) 

        residuals =  tables[n]['flux'] - median_pred_interp

        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='o', color='C' + str(n), alpha=0.5, **kwargs)
        ax[1].grid(True)

def plot_map_model_and_residuals(ax, data, pm_model, map_point, t_grid, 
        prediction, **kwargs):
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
    data : caustic.data 
        Microlensing event data. 
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    map_point : dict 
        Point in the parameter space for which we want to evaluate the 
        prediction tensor.
    t_grid : theano.tensor
        Times at which we want to evaluate model predictions. Shape 
        (n_bands, n_pts).
    prediction : theano.tensor
        Model prediction evaluated at t_grid.
    n_samples: int
        Number of posterior draws to be plotted.
    """
    # Load standardized data
    tables = data.get_standardized_data()

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()
    n_bands = len(data.light_curves)

    prediction_eval = np.zeros((n_bands, n_pts_dense))

    with pm_model:
        prediction_eval = xo.eval_in_model(prediction, map_point)

    # Plot model predictions for each different samples from posterior on dense 
    # grid 
    for n in range(n_bands): # iterate over bands
        ax[0].plot(t_grid[n].eval(), prediction_eval[n, :],
            color='C' + str(n), **kwargs)

    # Plot data
    data.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')
                
    # Compute residuals with respect to median model 
    for n in range(n_bands): 
        # Interpolate median predictions onto a grid of observed times
        map_prediction_interp = np.interp(tables[n]['HJD'], 
            t_grid[n].eval(), prediction_eval[n]) 

        residuals =  tables[n]['flux'] - map_prediction_interp

        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='o', color='C' + str(n), alpha=0.5, **kwargs)
        ax[1].grid(True)

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

def plot_trajectory(ax, pm_model, trace, n_samples=50, time_span=None,
        **kwargs):
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
    trace : PyMC3 MultiTrace object
        Trace object containing samples from posterior.
    time_span : list
        Timespan for which the model is to be evaluated [t_begin, t_end].
    n_samples: int
        Number of posterior draws to be plotted.
    """
    # Compute model predictions on a fine grid 
    with pm_model as m:
        if time_span is None:
            time_span = [m.t_begin, m.t_end]

        t_grid = np.linspace(time_span[0], time_span[1])

        pred = m.evaluate_trajectory_on_grid(trace, t_grid, 
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

def remove_outliers(pm_model, event):
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
    with pm_model:
        start = pm_model.test_point
        map_soln = xo.optimize(start=start, vars=[pm_model.F_base])
        map_soln = xo.optimize(start=map_soln, vars=[pm_model.Delta_F])
        map_soln = xo.optimize(start=map_soln, vars=[pm_model.u0])
        map_soln = xo.optimize(start=map_soln, vars=[pm_model.t0])
        map_soln = xo.optimize(start=map_soln, vars=[pm_model.u0, pm_model.teff])
        map_soln = xo.optimize(start=map_soln)

        t_observed = [T.as_tensor_variable(table['HJD']) for table in event.tables]
        pred_at_observed_times = pm_model.evaluate_map_model_on_grid(
            t_observed, map_soln)

    # Reomove all points deviating from MAP model by x*sigma_MAD
    for n in range(pm_model.n_bands):
        resid = event.tables[n]['flux'] - pred_at_observed_times[n]
        mad = lambda x: 1.4826*np.median(np.abs(x - np.median(x)))
        mask = np.abs(resid/mad(resid)) < 7

        # Updata mask
        event.light_curves[n]['mask'] = mask

    # Plot data 
    fig, ax = plt.subplots(figsize=(25, 10))
    event.plot(ax)