import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import pymc3 as pm
import seaborn as sns
import os
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

def solve_for_invgamma_params(params, t_min, t_max):
    def inverse_gamma_cdf(x, alpha, beta):
        return invgamma.cdf(x, alpha, scale=beta)

    alpha, beta = params

    return (inverse_gamma_cdf(2*t_min, alpha, beta) - \
    0.001, inverse_gamma_cdf(t_max, alpha, beta) - 0.99)

def calculate_likelihood(t, F, sigF, params):
    ln_sigma, ln_rho, DeltaF, Fb, t0, teff, tE = params

    # Set up mean model
    u0 = teff/tE
    u = theano.sqrt(u0**2 + ((t - t0)/tE)**2)
        
    A = lambda u: (u**2 + 2)/(u*theano.sqrt(u**2 + 4))  
    
    mean_model = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

    kernel = terms.Matern32Term(sigma=ln_sigma, rho=ln_rho)

    loglike, z, d, W, a, U, V, P = log_likelihood(kernel, 
        T.constant(1.4)*sigF, t, F)

    return loglike

# Define a theano Op for a custom likelihood function
class LogLike(T.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [T.dvector] # expects a vector of parameter values when called
    otypes = [T.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, t, F, sigF):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        t:
            The dependent variable (aka 't') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = calculate_likelihood

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs, t, F, sigF):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl = self.likelihood(t, F, sigF, theta)

        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values 
        theta, = inputs  # our parameters 
        return [g[0]*self.logpgrad(theta)]

class LogLikeGrad(T.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [T.dvector]
    otypes = [T.dvector]

    def __init__(self, loglike, gp, F):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # Get the gradients
        g = theano.function(inputs=[ln_s, ln_rho, DeltaF, Fb, t0, teff, tE], 
            outputs=theano.grad(self.loglike, 
            [ln_s, ln_rho,  DeltaF, Fb, t0, teff, tE]), 
                on_unused_input="ignore")

        outputs[0][0] = gradients


events = [] # event names
lightcurves = [] # data for each event
 
i = 0
n_events = 1
for entry in os.scandir('/home/star/fb90/data/OGLE_ews/2017/'):
    if entry.is_dir() and (i < n_events):
        events.append(entry.name)
        photometry = np.genfromtxt(entry.path + '/phot.dat', usecols=(0,1,2))
        lightcurves.append(photometry)
        i = i + 1
        
print("Loaded events:", events)

def fit_pymc3_model(t, F, sigF):
    model = pm.Model()

    # SPECIFICATION OF PRIORS
    # Compute parameters for the prior on GP hyperparameters
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

        ln_rho = pm.DensityDist('ln_rho', ln_rho_prior, testval = 0.6)

        def ln_sigma_prior(ln_sigma):
            sigma = np.exp(ln_sigma)
            res = np.log(sigma) - sigma**2/3.**2
            return T.cast(res, 'float64')

        ln_sigma = pm.DensityDist('ln_sigma', ln_sigma_prior, testval=2.)

        # Priors for unknown model parameters
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1.)
        Fb = pm.Normal('Fb', mu=0., sd=0.1)
        t0 = pm.Uniform('t0', 0, 1.)
        teff_tE = pm.DensityDist('teff_tE', joint_density, shape=2, 
            testval = [0.1, 10.])

        u_K = pm.Uniform('u_K', -1., 1.)

        K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))

        # Transform t0 to sensible units
        t0 = (t[-1] - t[0])*t0 + t[0]

        teff = teff_tE[0]
        tE = teff_tE[1]
        u0 = teff/tE

        # CALCULATE LIKELIHOOD
        def custom_log_likelihood(t, F, sigF):
            # Set up mean model
            u0 = teff/tE
            
            u = T.sqrt(u0**2 + ((t - t0)/tE)**2)

            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

            mean_function = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

            kernel = terms.Matern32Term(sigma=ln_sigma, rho=ln_rho)

            loglike = log_likelihood(kernel, mean_function,
                K*sigF, t, F)

            return loglike 

        logl = pm.DensityDist('logl', custom_log_likelihood, 
            observed={'t': t, 'F':F, 'sigF': sigF})

        # Initial parameters for the sampler
        t0_guess_idx = (np.abs(F - np.max(F))).argmin()

        # Initialization of the chain
        start = {'DeltaF':np.max(F), 'Fb':0.,
            't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}

        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point))

        # Fit model with NUTS
        trace = pm.sample(2000, tune=2000, nuts_kwargs=dict(target_accept=.95),
            start=start)

    return trace

for event_index, lightcurve in enumerate(lightcurves):

    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)


    # Fit pymc3 model
    trace = fit_pymc3_model(t, F, sigF)
    print(trace)