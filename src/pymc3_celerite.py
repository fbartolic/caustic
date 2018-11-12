import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import pymc3 as pm
import seaborn as sns
import os
import time
from codebase.data_preprocessing_ogle import process_data
from codebase.plotting_utils import plot_data

import theano
import theano.tensor as T
from theano.ifelse import ifelse

import sys
sys.path.append("theano_ops")
from theano_ops.celerite.factor import FactorOp
from theano_ops.celerite.solve import SolveOp
from theano_ops.celerite import terms
from theano_ops.celerite.celerite import log_likelihood

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

mpl.rc('text', usetex=False)
# DFM's pymc3 hack for estimating off-diagonal mass-matrix terms in NUTS during
# burn-in period
from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull
def get_step_for_trace(trace=None, model=None,
                       regular_window=5, regular_variance=1e-3,
                       **kwargs):
    model = pm.modelcontext(model)
    
    # If not given, use the trivial metric
    if trace is None:
        potential = QuadPotentialFull(np.eye(model.ndim))
        return pm.NUTS(potential=potential, **kwargs)
        
    # Loop over samples and convert to the relevant parameter space;
    # I'm sure that there's an easier way to do this, but I don't know
    # how to make something work in general...
    samples = np.empty((len(trace) * trace.nchains, model.ndim))
    i = 0
    for chain in trace._straces.values():
        for p in chain:
            samples[i] = model.bijection.map(p)
            i += 1
    
    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)
    
    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    # In the test case for this blog post, this isn't necessary and it
    # actually makes the performance worse so I'll disable it, but I
    # wanted to include the implementation here for completeness
    N = len(samples)
    cov = cov * N / (N + regular_window)
    cov[np.diag_indices_from(cov)] += \
        regular_variance * regular_window / (N + regular_window)
    
    # Use the sample covariance as the inverse metric
    potential = QuadPotentialFull(cov)
    return pm.NUTS(potential=potential, **kwargs)

def solve_for_invgamma_params(params, x_min, x_max):
    """Returns parameters of an inverse gamma distribution p(x) such that 
    0.1% of total prob. mass is assigned to values of x < x_min and 
    1% of total prob. masss  to values greater than x_max."""
    def inverse_gamma_cdf(x, alpha, beta):
        return invgamma.cdf(x, alpha, scale=beta)

    lower_mass = 0.01
    upper_mass = 0.99

    # Trial parameters
    alpha, beta = params

    # Equation for the roots defining params which satisfy the constraint
    return (inverse_gamma_cdf(x_min, alpha, beta) - \
    lower_mass, inverse_gamma_cdf(x_max, alpha, beta) - upper_mass)


# Define a tuning schedule for HMC
n_start = 25
n_burn = 500
n_tune = 5000
n_window = n_start * 2 ** np.arange(np.floor(np.log2((n_tune - n_burn) / n_start)))
n_window = np.append(n_window, n_tune - n_burn - np.sum(n_window))
n_window = n_window.astype(int)
np.random.seed(42)


def fit_pymc3_model(t, F, sigF):
    model = pm.Model()

    # Priors
    ## Compute parameters for the prior on GP hyperparameters
    invgamma_a, invgamma_b =  fsolve(solve_for_invgamma_params, (0.1, 0.1), 
        (np.median(np.diff(t)), t[-1] - t[0]))

    with model:    
        # log density of the joint prior for (tE, teff)
        def joint_density(value):
            teff = T.cast(value[0], 'float64')
            tE = T.cast(value[1], 'float64')
            sig_tE = T.cast(365., 'float64')
            sig_u0 = T.cast(1., 'float64')
            return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2

        # Priors for GP hyperparameters
        def ln_rho_prior(ln_rho):
            lnpdf_lninvgamma = lambda  x, a, b: np.log(x) + a*np.log(b) -\
                 (a + 1)*np.log(x) - b/x - np.log(gamma(a)) 

            res = lnpdf_lninvgamma(np.exp(ln_rho), invgamma_a, invgamma_b)
            return T.cast(res, 'float64')

        def ln_sigma_prior(ln_sigma):
            sigma = np.exp(ln_sigma)
            res = np.log(sigma) - sigma**2/3.**2
            return T.cast(res, 'float64')

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive

        # Model parameters
        ln_rho = pm.DensityDist('ln_rho', ln_rho_prior, testval = 0.6)
        ln_sigma = pm.DensityDist('ln_sigma', ln_sigma_prior, testval=2.)
        DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1., testval=3.)
        Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        t0 = pm.Uniform('t0', 0, 1.) 
        teff_tE = pm.DensityDist('teff_tE', joint_density, shape=2, 
            testval = [0.1, 10.])
       # u_K = pm.Uniform('u_K', -1., 1., testval=0.4)
       # K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))

        # Calculate likelihood
        def custom_log_likelihood(t, F, varF):
            # Set up mean model
            u0 = teff/tE
            
            u = T.sqrt(u0**2 + ((t - t0)/tE)**2)

            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

            mean_function = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

            kernel = terms.Matern32Term(sigma=T.exp(ln_sigma), rho=T.exp(ln_rho))

            loglike = log_likelihood(kernel, mean_function,
                varF, t, F)

            return loglike 

        ## Transform t0 to sensible units
        t0 = (t[-1] - t[0])*t0 + t[0]

        teff = teff_tE[0]
        tE = teff_tE[1]
        u0 = teff/tE

        logl = pm.DensityDist('logl', custom_log_likelihood, 
            observed={'t': t, 'F':F, 'varF': sigF**2})

        # Initial parameters for the sampler
        t0_guess_idx = (np.abs(F - np.max(F))).argmin()

        # Initialization of the chain
        start = {'DeltaF':np.max(F), 'Fb':0.,
            't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}

        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point))

        # Fit model with NUTS
        trace = pm.sample(200, tune=100, nuts_kwargs=dict(target_accept=.95),
         njobs=10, start=start)

        # DFM's optimized sampling procedure
#        burnin_trace = None
#        for steps in n_window:
#            step = get_step_for_trace(burnin_trace, regular_window=0)
#            burnin_trace = pm.sample(
#                start=start, tune=steps, draws=2, step=step,
#                compute_convergence_checks=False, discard_tuned_samples=False)
#            start = [t[-1] for t in burnin_trace._straces.values()]
#
#        step = get_step_for_trace(burnin_trace, regular_window=0)
#        dense_trace = pm.sample(draws=5000, tune=n_burn, step=step, start=start)
#        factor = 5000 / (5000 + np.sum(n_window+2) + n_burn)
#        dense_time = factor * (time.time() - strt)

    return trace

events = [] # event names
lightcurves = [] # data for each event
 
i = 0
n_events = 5
for entry in os.scandir('/home/star/fb90/data/OGLE_ews/2017/'):
    if entry.is_dir() and (i < n_events):
        events.append(entry.name)
        photometry = np.genfromtxt(entry.path + '/phot.dat', usecols=(0,1,2))
        lightcurves.append(photometry)
        i = i + 1
        
print("Loaded events:", events)

for event_index, lightcurve in enumerate(lightcurves):
    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

#    t = t[:1700]
#    F = F[:1700]
#    sigF = sigF[:1700]

    # Fit pymc3 model
    trace = fit_pymc3_model(t, F, sigF)
    #pm.summary(trace)
    #dense_time_per_eff = dense_time / stats.n_eff.min()
    #print("time per effective sample: {0:.5f} ms".format(dense_time_per_eff * 1000))

    # Save posterior samples
    samples_pymc3 = np.vstack([trace['DeltaF'],trace['Fb'],trace['t0'],
                         trace['teff_tE'][:, 0],trace['teff_tE'][:, 1],
                         trace['ln_sigma'], trace['ln_rho']]).T

    np.save('output/' + events[event_index] + '/samples_pymc3_celerite.npy', 
        samples_pymc3)

    # Save traceplots
    fig, ax = plt.subplots(6, 2 ,figsize=(10,10))
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig('output/' + events[event_index] + '/traceplots_celerite_pymc3.png')
