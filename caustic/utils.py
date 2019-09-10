import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import exoplanet as xo
import pymc3 as pm
import theano
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
    gp_list : list
        List of `exoplanet.gp.GP` objects, one per each band. If these
        are provided the likelihood which is computed is the GP marginal
        likelihood.
    """
    # Load standardized data
    tables = data.get_standardized_data()

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()
    n_bands = len(data.light_curves)

    prediction_eval = np.zeros((n_samples, n_bands, n_pts_dense))

    # Evaluate predictions in model context
    if gp_list==None:
        with pm_model:
            for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                    size=n_samples)):
                prediction_eval[i] = xo.eval_in_model(prediction, sample)

    else:
        with pm_model:
            prediction_tensors =\
                [gp_list[n].predict(t_grid[n]) for n in range(n_bands)]
            for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                    size=n_samples)):
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
        prediction, gp_list, **kwargs):
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
    gp_list : list
        List of `exoplanet.gp.GP` objects, one per each band. If these
        are provided the likelihood which is computed is the GP marginal
        likelihood.
    """
    # Load standardized data
    tables = data.get_standardized_data()

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

def plot_trajectory_from_samples(ax, data, pm_model, trace, t_grid, u_n, u_e, 
        n_samples=50, color='C0', **kwargs):
    """
    Plots the trajectory of the lens with respect to the source on the plane of
    the sky given samples from the posterior probability distribution. All 
    extra keyword arguments are passed to the matplotlib plot function.
    
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
    u_n : theano.tensor
        North component of the trajectory vector u(t).
    u_e : theano.tensor
        East component of the trajectory vector u(t).
    color : string
        Color of the plotted samples.
    n_samples: int
        Number of posterior draws to be plotted.
    """
    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()

    prediction_eval_u_n = np.zeros((n_samples, n_pts_dense))
    prediction_eval_u_e = np.zeros((n_samples, n_pts_dense))

    # Evaluate predictions in model context
    with pm_model:
        for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                size=n_samples)):
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

def plot_histograms_of_prior_samples(event, pm_model, output_dir):
    """
    Plots histograms of draws from prior distribution for each model parameter.
    The plots are saved to disk.
    
    Parameters
    ----------
    data : caustic.data
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

def compute_invgamma_params(data):
    """
    Returns parameters of an inverse zeta distribution p(x) such that 
    0.1% of total prob. mass is assigned to values of t < t_min and 
    1% of total prob. masss  to values greater than t_tmax. t_min is defined
    to be the median spacing between consecutive data points in the time series 
    and t_max is the total duration of the time series.
    
    Parameters
    ----------
    data : caustic.data 
        Microlensing event data. 
   
    Returns
    -------
    tuple
        (inv_gamma_a, inv_gamma_b) where each of the parameters has shape 
        (n_bands, 1).
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

def sample_with_emcee(pm_model, n_walkers=50, n_samples=10000, start=None):
    """
    Samples the model posterior distribution using emcee.
            
    Parameters
    ----------
    output_dir : str
        Path to directory where the trace is to be saved, together with
        various diagnostic plots and information. If it is not defined 
        nothing will be saved.
    n_walkers: int, optional
        Number of walkers, by default 50.
    n_samples : int, optional
        Number of sampling steps, by default 10000.
    start: dict, optional
        Initial point in the parameter space at which the tuning steps
        start.

    Returns
    -------
    sampler object 
        emcee sampler object 
    """
    import emcee 

    # Print the names of the free parameters and the inital values
    # of their log-priors        
    free_parameters = [RV.name for RV in pm_model.basic_RVs]
    initial_logps = [RV.logp(pm_model.test_point) for RV in pm_model.basic_RVs]
    if np.any(np.isnan(initial_logps))==True:
        print("Prior distributions misspecified, check that the test\
            values are within the bounds of the prior.")

    print("Free parameters:\n", free_parameters)

    # DFM's hack for using emcee with PyMC3 models
    f = theano.function(pm_model.vars,[pm_model.logpt] + pm_model.deterministics)

    def log_prob_func(params):
        dct = pm_model.bijection.rmap(params[::-1])
        args = (dct[k.name] for k in pm_model.vars)
        results = f(*args)
        return tuple(results)

    # First we work out the shapes of all of the deterministic variables
    initial_params = pm.find_MAP()

    # If custom initial parameters are specified, update relevant parameters
    if start is not None:
        for key in initial_params.keys():
            if key in start.keys():
                initial_params[key] = np.array([start[key]])

    # For some reason, bijection.map flips the ordering of the variables
    # from that in self.vars, hence [::-1]
    vec = pm_model.bijection.map(initial_params)[::-1]
    initial_blobs = log_prob_func(vec)[1:]
    dtype = [(var.name, float, np.shape(b)) for var,
            b in zip(pm_model.deterministics, initial_blobs)]
    
    # Then sample as usual
    coords = vec + 1e-5 * np.random.randn(n_walkers, len(vec))
    nwalkers, ndim = coords.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func, 
        blobs_dtype=dtype)
    sampler.run_mcmc(coords, n_samples, progress=True)

    return sampler

def sample_with_dynesty(pm_model, prior_transform, sampler_kwargs={}, 
        run_sampler_kwargs={}):
    """
    Samples the posterior distribution of a PyMC3 model using dynamic nested
    sampling as implemented in dynesty. 

    Parameters
    ----------
    pm_model: pymc3.Model
        PyMC3 model object defining the model we want to sample using dynesty.
    prior_transform : function
        Dyensty samples in a prior space where all parameters are i.i.d within
        a a D-dimensional unit cube. For independent parameters, this would be 
        the product of the inverse cumulative distribution function (CDF) 
        associated with each parameter. The function `prior_transform` should
        take an array of these uniformly distributed prior parameters and 
        transform them according to the actual prior we'd like to use. See
        the `dynesty` documentation for more details.
    sampler_kwargs: dict
        Additional arguments passed to the `dynesty.DynamicNestedSampler` 
        method.
    run_sampler_kwargs: dict
        Additional arguments passed to the 
        `dynesty.DynamicNestedSampler.run_nested` method.

    Returns
    -------
    dynesty.results.Result
        Object containing results of the Nested Sampling run. 
    """
    import dynesty 

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

    return sampler.results

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