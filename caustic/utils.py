import exoplanet as xo
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
from pymc3.util import get_untransformed_name, is_transformed_name


def construct_masked_tensor(array_list):
    """
    Given a list of 1D numpy arrays, this function returns a theano tensor
    of shape ``(n_elements, n_max)`` where ``n_elements`` is the number of 
    arrays in the list, and ``n_max`` is the length of the largest array 
    in the list. The missing values are filled in with zeros and a mask is 
    returned together with the tensor. 

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
        if array.ndim != 1:
            raise ValueError(
                "Make sure that all of the arrays are 1D and\
            have the same dimension"
            )

    n_max = np.max([len(array) for array in array_list])
    tensor = T.as_tensor_variable(
        np.stack(
            [
                np.pad(
                    array,
                    (0, n_max - len(array)),
                    "constant",
                    constant_values=(0.0,),
                )
                for array in array_list
            ]
        )
    )

    masks_list = []

    for array in array_list:
        array = np.append(np.ones(len(array)), np.zeros(n_max - len(array)))
        masks_list.append(array)

    mask = T.as_tensor_variable(np.stack(masks_list).astype("int8"))

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
    fluxes = np.concatenate([table["flux"] for table in tables])
    times = np.concatenate([table["HJD"] for table in tables])

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
        fluxes.append(np.median(table["flux"]))

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
        times = np.array(table["HJD"])
        idx = (np.abs(times - t_0)).argmin()
        fluxes.append(np.median(table["flux"][idx - 5 : idx + 5]))

    # Make sure that the estimated peak flux is greater than baseline flux
    if np.any(estimate_baseline_flux(data) > np.array(fluxes)):
        return 1.1 * estimate_baseline_flux(data)

    return np.array(fluxes)


def compute_source_mag_and_blend_fraction(
    data, Delta_F, F_base, u_0, model=None
):
    """
    Converts flux parameters :math:`(\Delta F,  F_\mathrm{base})` to physically
    more relevant interesting quantities, the source  star brightness in 
    magnitudes and the blend ratio :math:`g=F_B/F_S`.
    
    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Microlensing event data. 
    Delta_F : theano.tensor
        Tensor of shape ``(n_bands)``.
    F_base : theano.tensor
        Tensor of shape ``(n_bands)``.
    u_0 : theano.tensor
        Lens--source separation at time :math:`t_0`.
    standardized : bool
        Wether or not the flux is standardized to unit std deviation and zero
        median. By default ``True``.
    model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    
    Returns
    -------
    tuple
        ``(m_source, g)``.
    """
    model = pm.modelcontext(model)

    if model.standardized_data is True:
        # Revert F_base and Delta_F to non-standardized units
        data.units = "fluxes"
        fluxes_median = np.zeros(len(data.light_curves))
        fluxes_std = np.zeros(len(data.light_curves))

        for i, table in enumerate(data.light_curves):
            mask = table["mask"]
            fluxes_median[i] = np.median(table["flux"][mask])
            fluxes_std[i] = np.std(table["flux"][mask])

        # Flux parameters to standard flux units
        Delta_F_ = T.as_tensor_variable(fluxes_std) * Delta_F
        F_base_ = T.as_tensor_variable(
            fluxes_std
        ) * F_base + T.as_tensor_variable(fluxes_median)
    else:
        Delta_F_ = Delta_F
        F_base_ = F_base

    # Calculate source flux and blend flux
    A_u0 = (u_0 ** 2 + 2) / (T.abs_(u_0) * T.sqrt(u_0 ** 2 + 4))

    F_S = Delta_F_ / (A_u0 - 1)
    F_B = F_base_ - F_S

    g = F_B / F_S

    # Convert fluxes to magnitudes
    zero_point = 22.0
    m_source = zero_point - 2.5 * T.log10(F_S)

    return m_source, g


def plot_model_and_residuals(
    ax,
    data,
    trace,
    t_grid,
    prediction,
    n_samples=50,
    gp_list=None,
    model=None,
    **kwargs,
):
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
    model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    """
    model = pm.modelcontext(model)

    # Check if trace is a PyMC3 object or a raw numpy array, extract samples
    if isinstance(trace, np.ndarray):
        trace = trace[np.random.randint(len(trace), size=n_samples)]
        #  Map parameters to a dictionary which can be evaluated in model context
        samples = [model.bijection.rmap(params[::-1]) for params in trace]
    else:
        samples = xo.get_samples_from_trace(trace, size=n_samples)

    # Load data
    if model.standardized_data is True:
        tables = data.get_standardized_data()
    else:
        tables = data.get_standardized_data(rescale=False)

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()
    n_bands = len(data.light_curves)

    prediction_eval = np.zeros((n_samples, n_bands, n_pts_dense))

    # Evaluate predictions in model context
    if gp_list is None:
        for i, sample in enumerate(samples):
            prediction_eval[i] = xo.eval_in_model(prediction, sample)

    else:
        prediction_tensors = [
            gp_list[n].predict(t_grid[n]) for n in range(n_bands)
        ]
        for i, sample in enumerate(samples):
            for n in range(n_bands):
                prediction_eval[i, n] = xo.eval_in_model(
                    prediction_tensors[n], sample
                )

            # Add mean model to GP prediction
            prediction_eval[i] += xo.eval_in_model(prediction, sample)

    # Plot model predictions for each different samples from posterior on dense
    # grid
    for n in range(n_bands):  # iterate over bands
        for i in range(n_samples):
            ax[0].plot(
                t_grid[n].eval(),
                prediction_eval[i, n, :],
                color="C" + str(n),
                alpha=0.2,
                **kwargs,
            )

    # Compute median of the predictions
    median_predictions = np.zeros((n_bands, n_pts_dense))
    for n in range(n_bands):
        median_predictions[n] = np.percentile(
            prediction_eval[n], [16, 50, 84], axis=0
        )[1]

    # Plot data
    data.plot_standardized_data(ax[0], rescale=model.standardized_data)
    ax[0].set_xlabel(None)
    ax[1].set_xlabel("HJD - 2450000")
    ax[1].set_ylabel("Residuals")

    # Compute residuals with respect to median model
    for n in range(n_bands):
        # Interpolate median predictions onto a grid of observed times
        median_pred_interp = np.interp(
            tables[n]["HJD"], t_grid[n].eval(), median_predictions[n]
        )

        residuals = tables[n]["flux"] - median_pred_interp

        ax[1].errorbar(
            tables[n]["HJD"],
            residuals,
            tables[n]["flux_err"],
            fmt="o",
            color="C" + str(n),
            alpha=0.5,
            **kwargs,
        )
        ax[1].grid(True)


def plot_map_model_and_residuals(
    ax, data, map_point, t_grid, prediction, gp_list=None, model=None, **kwargs
):
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
    model : pymc3.Model
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
    model = pm.modelcontext(model)

    # Load data
    if model.standardized_data is True:
        tables = data.get_standardized_data()
    else:
        tables = data.get_standardized_data(rescale=False)

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()
    n_bands = len(data.light_curves)

    prediction_eval = np.zeros((n_bands, n_pts_dense))

    if gp_list is None:
        with model:
            prediction_eval = xo.eval_in_model(prediction, map_point)

    else:
        with model:
            for n in range(n_bands):
                prediction_eval[n] = xo.eval_in_model(
                    gp_list[n].predict(t_grid[n]), map_point
                )

            # Add mean model to GP prediction
            prediction_eval += xo.eval_in_model(prediction, map_point)

    # Plot model predictions for each different samples from posterior on dense
    # grid
    for n in range(n_bands):  # iterate over bands
        ax[0].plot(
            t_grid[n].eval(),
            prediction_eval[n, :],
            color="C" + str(n),
            **kwargs,
        )

    # Plot data
    data.plot_standardized_data(ax[0], rescale=model.standardized_data)
    ax[0].set_xlabel(None)
    ax[1].set_xlabel("HJD - 2450000")
    ax[1].set_ylabel("Residuals")

    # Compute residuals with respect to median model
    for n in range(n_bands):
        # Interpolate median predictions onto a grid of observed times
        map_prediction_interp = np.interp(
            tables[n]["HJD"], t_grid[n].eval(), prediction_eval[n]
        )

        residuals = tables[n]["flux"] - map_prediction_interp

        ax[1].errorbar(
            tables[n]["HJD"],
            residuals,
            tables[n]["flux_err"],
            fmt="o",
            color="C" + str(n),
            alpha=0.5,
            **kwargs,
        )
        ax[1].grid(True)


def plot_trajectory_from_samples(
    ax, trace, t_grid, u_n, u_e, n_samples=50, color="C0", model=None, **kwargs
):
    """
    Plots the trajectory of the lens with respect to the source on the plane of
    the sky given samples from the posterior probability distribution. All 
    extra keyword arguments are passed to the matplotlib plot function.
    
    Parameters
    ----------
    ax : matplotlib.axes 
        Needs to be of shape ``(2, 1)``.
    trace : PyMC3 MultiTrace object or ndarray
        Trace object containing samples from posterior. Assumed to be either
        a PyMC3 MultiTrace object, or a numpy array of shape 
        ``(n_samples, n_vars)`` containing raw samples in the transformed
        parameter space (without deterministic variables).
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
    model : pymc3.Model
        PyMC3 model object which was used to obtain posterior samples in the
        trace.
    """
    # Check if trace is a PyMC3 object or a raw numpy array, extract samples
    if isinstance(trace, np.ndarray):
        trace = trace[np.random.randint(len(trace), size=n_samples)]
        #  Map parameters to a dictionary which can be evaluated in model context
        samples = [model.bijection.rmap(params[::-1]) for params in trace]
    else:
        samples = xo.get_samples_from_trace(trace, size=n_samples)

    # Evaluate model for each sample on a fine grid
    n_pts_dense = T.shape(t_grid)[1].eval()

    prediction_eval_u_n = np.zeros((n_samples, n_pts_dense))
    prediction_eval_u_e = np.zeros((n_samples, n_pts_dense))

    # Evaluate predictions in model context
    for i, sample in enumerate(samples):
        prediction_eval_u_n[i] = xo.eval_in_model(u_n, sample)
        prediction_eval_u_e[i] = xo.eval_in_model(u_e, sample)

    ax.set_xlabel(r"$u_e\,[\theta_E]$")
    ax.set_ylabel(r"$u_n\,[\theta_E]$")
    ax.axhline(0.0, color="grey")
    ax.axvline(0.0, color="grey")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Source star
    circle = plt.Circle((0.0, 0.0), 0.1, color="C1", zorder=2)
    ax.add_artist(circle)

    ax.grid()

    # Plot predictions for various samples
    for i in range(n_samples):
        ax.plot(
            prediction_eval_u_e[i],
            prediction_eval_u_n[i],
            alpha=0.2,
            color=color,
            **kwargs,
        )


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
        return (
            invgamma.cdf(x_min, alpha, scale=beta) - lower_mass,
            invgamma.cdf(x_max, alpha, scale=beta) - upper_mass,
        )

    # Compute parameters for the prior on GP hyperparameters
    n_bands = len(data.light_curves)
    invgamma_a = np.zeros(n_bands)
    invgamma_b = np.zeros(n_bands)

    for i in range(n_bands):
        t_ = data.light_curves[i]["HJD"] - 2450000
        invgamma_a[i], invgamma_b[i] = fsolve(
            solve_for_params,
            (0.1, 0.1),
            (np.median(np.diff(t_)), t_[-1] - t_[0]),
        )

    return (
        T.as_tensor_variable(invgamma_a.reshape((n_bands, 1))),
        T.as_tensor_variable(invgamma_b.reshape((n_bands, 1))),
    )


def __get_untransformed_vars(model=None):
    """
    Get list of non-transformed RVs in the model.
    """
    model = pm.modelcontext(model)

    var_names = [str(var) for var in model.vars]
    for i, string in enumerate(var_names):
        if is_transformed_name(string):
            var_names[i] = get_untransformed_name(string)

    return [getattr(model, f"{name}") for name in var_names]


def get_log_likelihood_function(ll_tensor, model=None):
    """
    Builds a theano function from a PyMC3 model which takes a numpy array of
    shape ``(n_parameters)`` as an input  and returns returns the log
    likelihood of the model. The function takes the **untransformed** free
    parameters defined within the model context. The ordering of the parameters
    in the input array should match the ordering of the RVs in model context.
    The purpose of this function is to enable to use external samplers with
    PyMC3 models.
  
    Parameters
    ----------
    ll_tensor : theano.tensor
        Theano scalar which is the log likelihood of the model.
    model : pymc3.Model
        PyMC3 model object.
    
    Returns
    -------
    ndarray
        log likelihood of the model.
    """
    model = pm.modelcontext(model)

    variables = __get_untransformed_vars(model)
    f = theano.function(variables, [ll_tensor])

    def log_likelihood(params):
        # For some reason bijection.rmap switches the ordering, hence [::-1]
        dct = model.bijection.rmap(params[::-1])
        args = (dct[k.name] for k in model.vars)
        results = f(*args)
        return tuple(results)[0]

    return log_likelihood


def get_log_likelihood_function_grad(ll_tensor, model=None):
    """
    Same as :func:`caustic.utils.get_log_likelihood_function` except it returns
    the gradient of the log likelihood obtained with autodiff.
    """
    model = pm.modelcontext(model)

    variables = __get_untransformed_vars(model)
    grad = T.grad(ll_tensor, variables)
    f_grad = theano.function(variables, grad)

    def log_likelihood_grad(params):
        # For some reason bijection.rmap switches the ordering, hence [::-1]
        dct = model.bijection.rmap(params[::-1])
        args = (dct[k.name] for k in model.vars)
        results = f_grad(*args)
        return np.hstack([result.flatten() for result in results])

    return log_likelihood_grad


def get_log_probability_function(model=None):
    """
    Builds a theano function from a PyMC3 model which takes a numpy array of
    shape ``(n_parameters)`` as an input  and returns returns the total log
    probability of the model. This function takes the **transformed** random
    variables defined withing the model context which is a different behaviour
    from :func:`caustic.utils.get_log_likelihood_function`. The ordering of th
    para eters in the input array should match the ordering of the RVs in model 
    context. The purpose of this function is to be able to use external
    samplers with PyMC3 models. 
    
    Parameters
    ----------
    model : pymc3.Model
        PyMC3 model object.
    
    Returns
    -------
    ndarray
        Total log probability of the model.
    """
    model = pm.modelcontext(model)

    f = theano.function(model.vars, [model.logpt])

    def log_prob(params):
        dct = model.bijection.rmap(params[::-1])
        args = (dct[k.name] for k in model.vars)
        results = f(*args)
        return tuple(results)[0]

    return log_prob


def get_log_probability_function_grad(ll_tensor, model=None):
    """
    Same as :func:`caustic.utils.get_log_probability_function_grad` except it
    returns the gradient of the log probability obtained with autodiff.
    """
    model = pm.modelcontext(model)

    grad = T.grad(ll_tensor, model.vars)
    f_grad = theano.function(model.vars, grad)

    def log_probability_grad(params):
        # For some reason bijection.rmap switches the ordering, hence [::-1]
        dct = model.bijection.rmap(params[::-1])
        args = (dct[k.name] for k in model.vars)
        results = f_grad(*args)
        return np.hstack([result.flatten() for result in results])

    return log_probability_grad
