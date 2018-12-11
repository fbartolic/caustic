import numpy as np
import sys
sys.path.append("../../../exoplanet")
sys.path.append("../codebase")

import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import exoplanet as xo
from exoplanet.gp import terms, GP

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

print("PyMC3 version {0}".format(pm.__version__))

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
    return (inverse_gamma_cdf(10*x_min, alpha, beta) - \
    lower_mass, inverse_gamma_cdf(x_max, alpha, beta) - upper_mass)

def initialize_model(t, F, sigF):
    model = pm.Model()

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

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive

        # Noise model parameters
        ln_sigma = pm.DensityDist('ln_sigma', ln_sigma_prior, testval=np.log(2.))
        ln_rho = pm.DensityDist('ln_rho', ln_rho_prior, testval=np.log(10.))
        ln_K = pm.Normal('ln_K', mu=0., sd=1.5, testval=0.)
        K = T.exp(ln_K) + 1.
        #u_K = pm.Uniform('u_K', -1., 1.)
        #K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))

        # Microlensing rmodel parameters
        DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1., testval=3.)
        Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        t0 = pm.Uniform('t0', 0, 1., testval=0.5) 
        teff_tE = pm.DensityDist('teff_tE', joint_density, shape=2, 
            testval = [0.1, 10.])

        # Deterministic transformations
        teff = teff_tE[0]
        tE = teff_tE[1]
        u0 = pm.Deterministic("u0", teff/tE) # u0=teff/tE
        ## Transform t0 to sensible units
        t0 = (t[-1] - t[0])*t0 + t[0]

        # Calculate likelihood
        def mean_function(t):
            u = T.sqrt(u0**2 + ((t - t0)/tE)**2)
            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

            return DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

        mean = pm.Deterministic("flux", mean_function(t))
        
        kernel = terms.Matern32Term(log_sigma=ln_sigma, log_rho=ln_rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        gp = GP(kernel, t, (K*sigF)**2, J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", gp.log_likelihood(F - mean))
        pm.Deterministic("gp_pred", gp.predict())


        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point))

    return model, gp