import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import pymc3 as pm
import seaborn as sns
import os
import time

import theano
import theano.tensor as T
from theano.ifelse import ifelse

import sys
sys.path.append("../theano_ops")
sys.path.append("../codebase")
from data_preprocessing_ogle import process_data
from plotting_utils import *

from theano_ops.celerite.factor import FactorOp
from theano_ops.celerite.solve import SolveOp
from theano_ops.celerite import terms
from theano_ops.celerite.celerite import log_likelihood

import emcee
import celerite
from celerite.modeling import Model # an abstract class implementing the 
# skeleton of the celerite modeling protocol
from celerite import terms as celerite_terms 


from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

mpl.rc('text', usetex=False)

def solve_for_invgamma_params(params, x_min, x_max):
    """Returns parameters of an inverse gamma distribution p(x) such that 
    0.1% of total prob. mass is assigned to values of x < x_min and 
    1% of total prob. masss  to values greater than x_max."""
    def inverse_gamma_cdf(x, alpha, beta):
        return invgamma.cdf(x, alpha, scale=beta)

    lower_mass = 0.001
    upper_mass = 0.99

    # Trial parameters
    alpha, beta = params

    # Equation for the roots defining params which satisfy the constraint
    return (inverse_gamma_cdf(2*x_min, alpha, beta) - \
    lower_mass, inverse_gamma_cdf(x_max, alpha, beta) - upper_mass)

class GP_emcee(object):
    def __init__(self, t, F, sigF, *args, **kwargs):
        self.t = t
        self.F =  F
        self.sigF = sigF

        # Set up the GP model
        term1 = celerite_terms.Matern32Term(log_sigma=np.log(2.), log_rho=np.log(10))
        kernel = term1

        gp = celerite.GP(kernel)
        gp.compute(self.t, self.sigF)

        self.gp = gp

        print("Initial log-likelihood: {0}".format(gp.log_likelihood(F)))

        # Compute parameters for the prior on GP hyperparameters
        a, b =  fsolve(solve_for_invgamma_params, (0.1, 0.1), 
            (np.median(np.diff(self.t)), t[-1] - t[0]))
        #print("Median delta_t: ", np.diff(self.t))
        #print("Max t", t[-1] - t[0])
        self.rho_invgamma_params = [a, b]

    def log_prior(self, pars):

        lnL = 0 

        ln_sigma, ln_rho = pars

        # Prior for the GP hyperparameter
        lnpdf_lninvgamma = lambda  x, a, b: np.log(x) + a*np.log(b) -\
            (a + 1)*np.log(x) - b/x - np.log(gamma(a)) 

        # Variance parameter sigma
        sigma = np.exp(ln_sigma)
        lnL += np.log(sigma) - sigma**2/3.**2

        # Characteristic timescale rho 
        a = self.rho_invgamma_params[0]
        b = self.rho_invgamma_params[1]
        lnL += lnpdf_lninvgamma(np.exp(ln_rho), a, b)

        # Prior for u_K, the factor which rescales the errorbars
        #if u_K < -1 or u_K > 1:
        #    return -np.inf

        return lnL

    def log_posterior(self, pars):    
        ln_sigma, ln_rho = pars

        #if u_K < 0:
        #    K = 1.
        #else:
        #    K = 1 - np.log(1 - u_K)

        self.gp.compute(self.t, 1.4*self.sigF)
        self.gp.set_parameter_vector((ln_sigma,ln_rho))
        
        lp = self.log_prior(pars)
        
        if not np.isfinite(lp):
            return -np.inf
        return self.gp.log_likelihood(self.F) + lp 

    def sample(self, nsteps, nwalkers):
        initial_pars = self.gp.get_parameter_vector()
        #initial_pars = np.append(initial_pars_gp, (0,))

        ndim, nwalkers = len(initial_pars), nwalkers
        print("ndim:   ", ndim)
        p0 = initial_pars + 1e-8 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()

        print("Running production...")
        sampler.run_mcmc(p0, nsteps);
        
        return sampler, self.gp

def fit_pymc3_model(t, F, sigF):
    model = pm.Model()

    # SPECIFICATION OF PRIORS
    # Compute parameters for the prior on GP hyperparameters
    invgamma_a, invgamma_b =  fsolve(solve_for_invgamma_params, (0.1, 0.1), 
        (np.median(np.diff(t)), t[-1] - t[0]))

    with model:    
        # Priors
        ## log density of the joint prior for (tE, teff)
        def joint_density(value):
            teff = T.cast(value[0], 'float64')
            tE = T.cast(value[1], 'float64')
            sig_tE = T.cast(365., 'float64')
            sig_u0 = T.cast(1., 'float64')
            return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2

        ## priors for GP hyperparameters
        def ln_rho_prior(ln_rho):
            lnpdf_lninvgamma = lambda  x, a, b: np.log(x) + a*np.log(b) -\
                 (a + 1)*np.log(x) - b/x - np.log(gamma(a)) 

            res = lnpdf_lninvgamma(np.exp(ln_rho), invgamma_a, invgamma_b)
            return T.cast(res, 'float64')

        def ln_sigma_prior(ln_sigma):
            sigma = np.exp(ln_sigma)
            res = np.log(sigma) - sigma**2/3.**2
            return T.cast(res, 'float64')

        BoundedNormal = pm.Bound(pm.Normal, lower=1.) 

        # Parameters
        ln_sigma = pm.DensityDist('ln_sigma', ln_sigma_prior, testval=2.)
        ln_rho = pm.DensityDist('ln_rho', ln_rho_prior, testval=0.6)
        lnK = pm.Normal('lnK', mu=0., sd=1., testval=0.)
        #u_K = pm.Uniform('u_K', -1., 1.)
        #K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))

        # Custom likelihood function calling celerite
        def custom_log_likelihood(t, F, sigF):
            kernel = terms.Matern32Term(sigma=T.exp(ln_sigma), 
                rho=T.exp(ln_rho))

            loglike = log_likelihood(kernel, 0.,
                (T.exp(lnK) + 1.)*sigF, t, F)

            return loglike 

        logl = pm.DensityDist('logl', custom_log_likelihood, 
            observed={'t': t, 'F':F, 'sigF': sigF})

        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point))

        # Fit model with NUTS
        trace = pm.sample(2000, tune=2000, nuts_kwargs=dict(target_accept=.95))

    return trace

limits = [1700, 1900, 1900, 1700, 2500]

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

    t = t[:limits[event_index]]
    F = F[:limits[event_index]]
    sigF = sigF[:limits[event_index]]

    # Fit pymc3 model
    trace = fit_pymc3_model(t, F, sigF)

    # Fit emcee model
    #model_emcee_GP = GP_emcee(t, F, sigF)
    #sampler_emcee_GP, gp = model_emcee_GP.sample(2000., 20)

    # Save posterior samples
    samples_pymc3 = pm.trace_to_dataframe(trace, 
        varnames=["ln_sigma", "ln_rho"]).values.T


    #samples_pymc3 = np.vstack([trace['ln_sigma'], trace['ln_rho']])
    #samples_emcee_GP = sampler_emcee_GP.chain

    #np.save('samples_emcee.npy', samples_emcee_GP)
    np.save( events[event_index] + '_samples_pymc3.npy', samples_pymc3)

    # Save traceplots
    fig, ax = plt.subplots(3, 2 ,figsize=(10,5))
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig(events[event_index] + 'traceplots_celerite_pymc3.png')

