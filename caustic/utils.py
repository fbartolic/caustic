import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os

import exoplanet as xo
import pymc3 as pm
import theano
import theano.tensor as T


import caustic as ca


def construct_masked_tensor(array_list):
    """
    Given a list of 1D numpy arrays, this function returns a theano tensor
    of shape ``(n_elements, n_max)`` where ``n_elements`` is the number of 
    arrays in the list, and ``n_max`` is the length of the largest array 
    in the list. The missing values are filled in with zeros and a mask is returned 
    together with the tensor. 

    Parameters
    ----------
    array_list : list 
        List of numpy arrays of varying lengths. 
    
    Returns
    -------
    tuple 
        Returns tuple ``(tensor, mask)`` where tensor and maks are 
        ``theano.tensor`` objects of the same shape ``(n_elements, n_max)``. 
        The mask tensor is of datta type ``int8`` and the elements are 
        equal to 1 for non-filled in values and zero otherwise. To use
        the mask in theano, use ``tensor[mask.nonzero()]``.

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


def estimate_t0(data):
    """
    Estimates the initial value for the :math:`t_0` parameter. This is 
    necessary because the posterior is highly multi-modal in :math:`t_0` 
    and the sampler usually fails to converge if :math:`t_0` is not close 
    to true value.

    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Microlensing event data. 

    Returns
    -------
    float
        Estimate of :math:`t_0`.

    """
    tables = data.get_standardized_data()
    fluxes = np.concatenate([table['flux'] for table in tables])
    times = np.concatenate([table['HJD'] for table in tables])

    guess = np.median(times[fluxes > 4])

    cut = 4
    while np.isnan(guess):
        guess = np.median(times[fluxes > cut])
        cut -= 0.5

    return guess


def estimate_baseline_flux(data):
    """
    Estimator for the baseline flux of an event.
    
    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Microlensing event data. 

    Returns
    -------
    ndarray 
        Estimate of baseline flux in each band.
    """
    tables = data.get_standardized_data(rescale=False)

    fluxes = []

    for table in tables:
        fluxes.append(np.median(table['flux']))

    return np.array(fluxes)


def estimate_peak_flux(data):
    """
    Estimator for peak flux of light curves.

    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Microlensing event data. 

    Returns
    -------
    ndarray 
        Estimate of peak flux in each band.
    """
    tables = data.get_standardized_data(rescale=False)

    fluxes = []

    t_0 = estimate_t0(data)

    for table in tables:
        times = np.array(table['HJD'])
        idx = (np.abs(times - t_0)).argmin()
        fluxes.append(np.median(table['flux'][idx-5:idx+5]))

    # Make sure that the estimated peak flux is greater than baseline flux
    if np.any(estimate_baseline_flux(data) > np.array(fluxes)):
        return 1.1*estimate_baseline_flux(data) 

    return np.array(fluxes)


def compute_source_mag_and_blend_fraction(data, pm_model, Delta_F, F_base, u_0):
    """
    Converts flux parameters :math:`(\Delta F,  F_\mathrm{base})` to physically
    more relevant interesting quantities, the source  star brightness in 
    magnitudes and the blend ratio :math:`g=F_B/F_S`.
    
    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Microlensing event data. 
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    Delta_F : theano.tensor
        Tensor of shape ``(n_bands)``.
    F_base : theano.tensor
        Tensor of shape ``(n_bands)``.
    u_0 : theano.tensor
        Lens--source separation at time :math:`t_0`.
    standardized : bool
        Wether or not the flux is standardized to unit std deviation and zero
        median. By default ``True``.
    
    Returns
    -------
    tuple
        ``(m_source, g)``.
    """
    if pm_model.standardized_data==True:
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
    else:
        Delta_F_ = Delta_F
        F_base_ = F_base

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
        n_samples=50, gp_list=None, **kwargs):
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
        Needs to be of shape ``(2, 1)``.
    data : :func:`~caustic.data.Data`
        Microlensing event data. 
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    trace : PyMC3 MultiTrace object or ndarray
        Trace object containing samples from posterior. Assumed to be either
        a PyMC3 MultiTrace object, or a numpy array of shape 
        ``(n_samples, n_vars)`` containing raw samples in the transformed 
        parameter space (without deterministic variables).
    t_grid : theano.tensor
        Times at which we want to evaluate model predictions. Shape 
        ``(n_bands, n_pts)``.
    prediction : theano.tensor
        Model prediction evaluated at ``t_grid``.
    n_samples: int
        Number of posterior draws to be plotted.
    gp_list : list
        List of ``exoplanet.gp.GP`` objects, one per each band. If these
        are provided the likelihood which is computed is the GP marginal
        likelihood.
    """
    # Check if trace is a PyMC3 object or a raw numpy array, extract samples
    if isinstance(trace, np.ndarray):
        trace = trace[np.random.randint(len(trace), size=n_samples)]
        # Map parameters to a dictionary which can be evaluated in model context
        samples = [pm_model.bijection.rmap(params[::-1]) for params in trace]
    else: 
        samples = xo.get_samples_from_trace(trace, size=n_samples)

    # Load data
    if pm_model.standardized_data==True:
        tables = data.get_standardized_data()
    else:
        tables = data.get_standardized_data(rescale=False)

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()
    n_bands = len(data.light_curves)

    prediction_eval = np.zeros((n_samples, n_bands, n_pts_dense))

    # Evaluate predictions in model context
    if gp_list==None:
        with pm_model:
            for i, sample in enumerate(samples):
                prediction_eval[i] = xo.eval_in_model(prediction, sample)

    else:
        with pm_model:
            prediction_tensors =\
                [gp_list[n].predict(t_grid[n]) for n in range(n_bands)]
            for i, sample in enumerate(samples):
                for n in range(n_bands):
                    prediction_eval[i, n] =\
                        xo.eval_in_model(prediction_tensors[n], sample) 

                # Add mean model to GP prediction
                prediction_eval[i] += xo.eval_in_model(prediction, sample)

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
    data.plot_standardized_data(ax[0], rescale=pm_model.standardized_data)
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
        prediction, gp_list=None, **kwargs):
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
        Needs to be of shape ``(2, 1)``.
    data : :func:`~caustic.data.Data`
        Microlensing event data. 
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    map_point : dict 
        Point in the parameter space for which we want to evaluate the 
        prediction tensor.
    t_grid : theano.tensor
        Times at which we want to evaluate model predictions. Shape 
        ``(n_bands, n_pts)``.
    prediction : theano.tensor
        Model prediction evaluated at ``t_grid``.
    gp_list : list
        List of ``exoplanet.gp.GP`` objects, one per each band. If these
        are provided the likelihood which is computed is the GP marginal
        likelihood.
    """
    # Load data
    if pm_model.standardized_data==True:
        tables = data.get_standardized_data()
    else:
        tables = data.get_standardized_data(rescale=False)

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()
    n_bands = len(data.light_curves)

    prediction_eval = np.zeros((n_bands, n_pts_dense))

    if gp_list==None:
        with pm_model:
            prediction_eval = xo.eval_in_model(prediction, map_point)

    else:
        with pm_model:
            for n in range(n_bands):
                prediction_eval[n] =\
                    xo.eval_in_model(gp_list[n].predict(t_grid[n]), map_point) 

            # Add mean model to GP prediction
            prediction_eval += xo.eval_in_model(prediction, map_point)

    # Plot model predictions for each different samples from posterior on dense 
    # grid 
    for n in range(n_bands): # iterate over bands
        ax[0].plot(t_grid[n].eval(), prediction_eval[n, :],
            color='C' + str(n), **kwargs)

    # Plot data
    data.plot_standardized_data(ax[0], rescale=pm_model.standardized_data)
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


def plot_trajectory_from_samples(ax, pm_model, trace, t_grid, u_n, u_e, 
        n_samples=50, color='C0', **kwargs):
    """
    Plots the trajectory of the lens with respect to the source on the plane of
    the sky given samples from the posterior probability distribution. All 
    extra keyword arguments are passed to the matplotlib plot function.
    
    Parameters
    ----------
    ax : matplotlib.axes 
        Needs to be of shape ``(2, 1)``.
    pm_model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    trace : PyMC3 MultiTrace object or ndarray
        Trace object containing samples from posterior. Assumed to be either
        a PyMC3 MultiTrace object, or a numpy array of shape 
        ``(n_samples, n_vars)`` containing raw samples in the transformed parameter
        space (without deterministic variables).
    t_grid : theano.tensor
        Times at which we want to evaluate model predictions. Shape 
        ``(n_bands, n_pts)``.
    u_n : theano.tensor
        North component of the trajectory vector :math:`\boldsymbol{u}(t)`.
    u_e : theano.tensor
        East component of the trajectory vector :math:`\boldsymbol{u}(t)`.
    color : string
        Color of the plotted samples.
    n_samples: int
        Number of posterior draws to be plotted.
    """
    # Check if trace is a PyMC3 object or a raw numpy array, extract samples
    if isinstance(trace, np.ndarray):
        trace = trace[np.random.randint(len(trace), size=n_samples)]
        # Map parameters to a dictionary which can be evaluated in model context
        samples = [pm_model.bijection.rmap(params[::-1]) for params in trace]
    else: 
        samples = xo.get_samples_from_trace(trace, size=n_samples)

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()

    prediction_eval_u_n = np.zeros((n_samples, n_pts_dense))
    prediction_eval_u_e = np.zeros((n_samples, n_pts_dense))

    # Evaluate predictions in model context
    with pm_model:
        for i, sample in enumerate(samples):
            prediction_eval_u_n[i] = xo.eval_in_model(u_n, sample)
            prediction_eval_u_e[i] = xo.eval_in_model(u_e, sample)

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
        ax.plot(prediction_eval_u_e[i], prediction_eval_u_n[i], alpha=0.2, 
            color=color, **kwargs)


def compute_invgamma_params(data):
    """
    Returns parameters of an inverse zeta distribution :math:`p(x)` such that 
    0.1% of total prob. mass is assigned to values of :math:`t < t_{min}` and 
    1% of total prob. masss  to values greater than :math:`t_{tmax}`. 
    :math:`t_{min}` is defined to be the median spacing between consecutive 
    data points in the time series and :math:`t_{max}` is the total duration 
    of the time series.
    
    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Microlensing event data. 
   
    Returns
    -------
    tuple
        ``(inv_gamma_a, inv_gamma_b)`` where each of the parameters has shape 
        ``(n_bands, 1)``.
    """
    from scipy.stats import invgamma
    from scipy.optimize import fsolve

    def solve_for_params(params, x_min, x_max):
        lower_mass = 0.01
        upper_mass = 0.99

        # Trial parameters
        alpha, beta = params

        # Equation for the roots defining params which satisfy the constraint
        return (invgamma.cdf(x_min, alpha, scale=beta) - \
            lower_mass, invgamma.cdf(x_max, alpha, scale=beta) - upper_mass)

    # Compute parameters for the prior on GP hyperparameters
    n_bands = len(data.light_curves)
    invgamma_a = np.zeros(n_bands)
    invgamma_b = np.zeros(n_bands)

    for i in range(n_bands):
        t_ = data.light_curves[i]['HJD'] - 2450000
        invgamma_a[i], invgamma_b[i] = fsolve(solve_for_params, 
            (0.1, 0.1), (np.median(np.diff(t_)), t_[-1] - t_[0]))

    return (T.as_tensor_variable(invgamma_a.reshape((n_bands, 1))),
        T.as_tensor_variable(invgamma_b.reshape((n_bands, 1))))


def sample_with_dynesty(pm_model, prior_transform, sampler_kwargs={}, 
        run_sampler_kwargs={}):
    """
    Samples the posterior distribution of a ``PyMC3`` model using dynamic nested
    sampling as implemented in ``dynesty``. 

    Parameters
    ----------
    pm_model: pymc3.Model
        PyMC3 model object defining the model we want to sample using ``dynesty``.
    prior_transform : function
        Dyensty samples in a prior space where all parameters are i.i.d within
        a a D-dimensional unit cube. For independent parameters, this would be 
        the product of the inverse cumulative distribution function (CDF) 
        associated with each parameter. The function should
        take an array of these uniformly distributed prior parameters and 
        transform them according to the actual prior we'd like to use. See
        the ``dynesty`` documentation for more details.
    sampler_kwargs: dict
        Additional arguments passed to the :code:`dynesty.DynamicNestedSampler` 
        method.
    run_sampler_kwargs: dict
        Additional arguments passed to the 
        :code:`dynesty.DynamicNestedSampler.run_nested` method.

    Returns
    -------
    tuple 
        Returns a tuple of type ``(dynesty.results.Result, ndarray)`` with the 
        object containing results of the Nested Sampling run and a numpy array
        with reweighted posterior samples.
    """
    import dynesty 
    from dynesty import utils as dyfunc

    with pm_model:              
        print("Model vars", pm_model.vars)

        # Builds a theano function which takes the model parameters
        # and returns the log likelihood
        f = theano.function(pm_model.vars, [pm_model.potentials[0]])
        
        def log_likelihood(params):
            dct = pm_model.bijection.rmap(params[::-1])
            args = (dct[k.name] for k in pm_model.vars)
            results = f(*args)
            return tuple(results)[0]
        
        ndim = len(pm_model.vars)
            
        sampler = dynesty.DynamicNestedSampler(log_likelihood, 
            prior_transform, ndim, **sampler_kwargs)
        sampler.run_nested(wt_kwargs={'pfrac': 1.0}, print_progress=True, 
            **run_sampler_kwargs)

    results = sampler.results 

    # Resample samples such that they have equal weight
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    new_samples = dyfunc.resample_equal(samples, weights)

    return sampler.results, new_samples