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

class PointSourcePointLens(pm.Model):
    #  override __init__ function from pymc3 Model class
    def __init__(self, t, F, sigF, name='', model=None):
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super(PointSourcePointLens, self).__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        # Data
        self.t = t
        self.F = F
        self.sigF = sigF

        # Custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(F - np.max(F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=t[t0_guess_idx])
        self.teff_tE = pm.DensityDist('teff_tE', self.joint_density, shape=2, 
            testval = [35., 55.])

        # Deterministic transformations
        teff = self.teff_tE[0]
        self.tE = self.teff_tE[1]
        self.u0 = pm.Deterministic("u0", teff/self.tE) # u0=teff/tE

        # Noise model parameters
        self.K = BoundedNormal1('K', mu=1.001, sd=2., testval=1.5)
        
        Y_obs = pm.Normal('Y_obs', mu=self.mean_function(), sd=self.K*sigF, 
            observed=F)

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def joint_density(self, value):
        teff = T.cast(value[0], 'float64')
        tE = T.cast(value[1], 'float64')
        sig_tE = T.cast(365., 'float64')
        sig_u0 = T.cast(1., 'float64')
        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2


class PointSourcePointLensMatern32(pm.Model):
    def __init__(self, t, F, sigF, name='', model=None):
        super(PointSourcePointLensMatern32, self).__init__(name, model)

        # Data
        self.t = t
        self.F = F
        self.sigF = sigF

        # Custom prior distributions 
        # Compute parameters for the prior on GP hyperparameters
        invgamma_a, invgamma_b =  fsolve(self.solve_for_invgamma_params, (0.1, 0.1), 
        (np.median(np.diff(self.t)), self.t[-1] - self.t[0]))

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(F - np.max(F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=t[t0_guess_idx])
        self.teff_tE = pm.DensityDist('teff_tE', self.joint_density, shape=2, 
            testval = [35., 55.])

        # Deterministic transformations
        teff = self.teff_tE[0]
        self.tE = self.teff_tE[1]
        self.u0 = pm.Deterministic("u0", teff/self.tE) # u0=teff/tE

        # Noise model parameters
        self.sigma = BoundedNormal('sigma', mu=0., sd=3., testval=0.5)
        self.rho = pm.InverseGamma('rho', alpha=invgamma_a, beta=invgamma_b, 
            testval=3.)
        self.K = BoundedNormal1('K', mu=1.001, sd=2., testval=1.5)
        self.ln_rho = pm.Deterministic('ln_rho', T.log(self.rho))
        self.ln_sigma = pm.Deterministic('ln_sigma', T.log(self.sigma))

        # Calculate likelihood
        kernel = terms.Matern32Term(sigma=self.sigma, rho=self.rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        self.gp = GP(kernel, t, (self.K*sigF)**2, J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", self.gp.log_likelihood(F - self.mean_function()))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def joint_density(self, value):
        teff = T.cast(value[0], 'float64')
        tE = T.cast(value[1], 'float64')
        sig_tE = T.cast(365., 'float64')
        sig_u0 = T.cast(1., 'float64')
        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2

    def solve_for_invgamma_params(self, params, x_min, x_max):
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
