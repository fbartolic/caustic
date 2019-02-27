import numpy as np
import sys

import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import exoplanet as xo
from exoplanet.gp import terms, GP
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

class PointSourcePointLens(pm.Model):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, name='', model=None):
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super(PointSourcePointLens, self).__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        # Load and pre-process the data 
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # Delta_F is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Define model parameters and their associated priors
        self.Delta_F = BoundedNormal('Delta_F', mu=np.max(self.F), sd=1., testval=3.)
        self.F_base = pm.Normal('F_base', mu=0., sd=0.1, testval=0.)

        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])
        self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        # Transform (Delta_F,F_basease) to (m_S,m_blend) using deterministic 
        # mapping and save to trace
        m_source, m_blend = self.revert_flux_params_to_nonstandardized_format(
            data)
        self.mag_source = pm.Deterministic("m_source", m_source)
        self.mag_blend = pm.Deterministic("m_blend", m_blend)

        # Noise model parameters
        self.A = BoundedNormal1('A', mu=1., sd=2., testval=1.5)
        self.B = BoundedNormal('B', mu=0., sd=2., testval=0.01)

        # Save log prior for each parameter, this is needed for hierarchical
        # modeling of multiple events using the importance resampling trick
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            BoundedNormal.dist(mu=np.max(self.F), sd=1.).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(mu=0., sd=0.1).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t[0], self.t[-1]).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            BoundedNormal.dist(mu=0., sd=1.).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))
        self.logp_A = pm.Deterministic('logp_A',
            BoundedNormal1.dist( mu=1., sd=2.).logp(self.A))
        self.logp_B = pm.Deterministic('logp_B',
            BoundedNormal1.dist( mu=0., sd=2.).logp(self.B))
        self.log_posterior = pm.Deterministic("log_posterior", self.logpt)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

        # Define the likelihood function
        mean = self.mean_function()
        sigF_modeled = T.sqrt(T.pow(self.A*T._shared(self.sigF), 2)+\
             T.pow(self.magnification()*self.B, 2))
        Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigF_modeled, 
            observed=self.F, shape=len(self.F))

    def magnification(self):
        """Return the mean function which goes into the likeliood."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 

    def mean_function(self):
        """Return the mean function which goes into the likeliood."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.Delta_F*(A(u) - 1)/(A(self.u0) - 1) + self.F_base

    def revert_flux_params_to_nonstandardized_format(self, data):
        # Revert F_base and Delta_F to non-standardized units
        median_F = np.median(data.df['I_flux'].values)
        std_F = np.std(data.df['I_flux'].values)

        Delta_F_ = std_F*self.Delta_F + median_F
        F_base_ = std_F*self.Delta_F + median_F

        # Calculate source flux and blend flux
        FS = Delta_F_/(self.peak_mag() - 1)
        FB = (F_base_ - FS)/FS

        # Convert fluxes to magnitudes
        mu_m, sig_m = data.fluxes_to_magnitudes(np.array([FS, FB]), 
            np.array([0., 0.]))
        mag_source, mag_blend = mu_m

        return mag_source, mag_blend

    def peak_mag(self):
        """Returns PSPL magnification at u=u0."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

class PointSourcePointLensMarginalized(pm.Model):
    def __init__(self, data, name='', model=None):
        super(PointSourcePointLensMarginalized, self).__init__(name, model)

        # Load and pre-process the data 
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Define model parameters and their associated priors
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])
        self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        # Noise model parameters
        self.A = BoundedNormal1('A', mu=1., sd=2., testval=1.5)
        self.B = BoundedNormal('B', mu=0., sd=2., testval=0.01)

        # Save log prior for each parameter, this is needed for hierarchical
        # modeling of multiple events using the importance resampling trick
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t[0], self.t[-1]).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            BoundedNormal.dist(mu=0., sd=1.).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))
        self.logp_A = pm.Deterministic('logp_A',
            BoundedNormal1.dist( mu=1., sd=2.).logp(self.A))
        self.logp_B = pm.Deterministic('logp_B',
            BoundedNormal1.dist( mu=0., sd=2.).logp(self.B))
        self.log_posterior = pm.Deterministic("log_posterior", self.logpt)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

        # Define the likelihood function~
        pm.Potential('likelihood', self.marginalized_likelihood())

    def marginalized_likelihood(self):
        """Gaussian likelihood funciton marginalized over the linear parameters.
        """
        N = len(self.F)        
        F = T._shared(self.F)

        # Linear parameter matrix
        mag_vector = self.magnification()
#        mu_theta = T.dot(mag_vector, np.max(self.F))
        A = T.stack([mag_vector, T.ones(N)], axis=1)

        # Covariance matrix
        C_diag = T.pow(self.A*T._shared(self.sigF), 2)+\
             T.pow(self.magnification()*self.B, 2)
        C = T.nlinalg.diag(C_diag)

        # Prior matrix
        sigDelta_F = 10.
        sigF_base = 0.1
        L_diag = T._shared(np.array([sigDelta_F, sigF_base])**2.)
        L = T.nlinalg.diag(L_diag)

        # Calculate inverse of covariance matrix for marginalized likelihood
        inv_C = T.nlinalg.diag(T.pow(C_diag, -1.))
        inv_L = T.nlinalg.diag(T.pow(L_diag, -1.))
        term1 = T.dot(A.transpose(), inv_C) 
        term2 = inv_L + T.dot(A.transpose(), T.dot(inv_C, A))
        term3 = T.dot(inv_C, A)
        inv_SIGMA = inv_C - T.dot(term3, T.dot(T.nlinalg.matrix_inverse(term2),
             term1))

        # Calculate determinant of covariance matrix for marginalized likelihood
        det_C = C_diag.prod() 
        det_L = L_diag.prod() 
        det_SIGMA = det_C*det_L*T.nlinalg.det(term2)

        # Calculate marginalized likelihood
        r = F #- mu_theta
        return -0.5*T.dot(r.transpose(), T.dot(inv_SIGMA, r)) -\
               0.5*N*np.log(2*np.pi) - 0.5*np.log(det_SIGMA)

    def magnification(self):
        """Return the mean function which goes into the likeliood."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 

class PointSourcePointLensMatern32(pm.Model):
    def __init__(self, data, name='', model=None):
        super(PointSourcePointLensMatern32, self).__init__(name, model)

        # Load and pre-process the data 
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # Delta_F is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Define model parameters and their associated priors
        self.Delta_F = BoundedNormal('Delta_F', mu=np.max(self.F), sd=1., testval=3.)
        self.F_base = pm.Normal('F_base', mu=0., sd=0.1, testval=0.)

        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])
        self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        # Transform (Delta_F,F_basease) to (m_S,m_blend) using deterministic 
        # mapping and save to trace
        m_source, m_blend = self.revert_flux_params_to_nonstandardized_format(
            data)
        self.mag_source = pm.Deterministic("m_source", m_source)
        self.mag_blend = pm.Deterministic("m_blend", m_blend)

        # Noise model parameters
        self.A = BoundedNormal1('A', mu=1., sd=2., testval=1.5)
        self.B = BoundedNormal('B', mu=0., sd=2., testval=0.01)
        ## Compute parameters for the prior on GP hyperparameters
        invgamma_a, invgamma_b =  fsolve(self.solve_for_invgamma_params, (0.1, 0.1), 
        (np.median(np.diff(self.t)), self.t[-1] - self.t[0]))

        self.sigma = BoundedNormal('sigma', mu=0., sd=3., testval=0.5)
        self.rho = pm.InverseGamma('rho', alpha=invgamma_a, beta=invgamma_b, 
            testval=3.)

        # Save log prior for each parameter, this is needed for hierarchical
        # modeling of multiple events using the importance resampling trick
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            BoundedNormal.dist(mu=np.max(self.F), sd=1.).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(mu=0., sd=0.1).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t[0], self.t[-1]).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            BoundedNormal.dist(mu=0., sd=1.).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))
        self.logp_A = pm.Deterministic('logp_A',
            BoundedNormal1.dist( mu=1., sd=2.).logp(self.A))
        self.logp_B = pm.Deterministic('logp_B',
            BoundedNormal1.dist( mu=0., sd=2.).logp(self.B))
        self.log_posterior = pm.Deterministic("log_posterior", self.logpt)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

        # Define the likelihood function
        mean = self.mean_function()
        varF_modeled = T.pow(self.A*T._shared(self.sigF), 2)+\
             T.pow(self.magnification()*self.B, 2)

        # Calculate likelihood
        kernel = terms.Matern32Term(sigma=self.sigma, rho=self.rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        self.gp = GP(kernel, self.t, varF_modeled, J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", self.gp.log_likelihood(self.F - self.mean_function()))

    def magnification(self):
        """Return the mean function which goes into the likeliood."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 

    def mean_function(self):
        """Return the mean function which goes into the likeliood."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.Delta_F*(A(u) - 1)/(A(self.u0) - 1) + self.F_base

    def revert_flux_params_to_nonstandardized_format(self, data):
        # Revert F_base and Delta_F to non-standardized units
        median_F = np.median(data.df['I_flux'].values)
        std_F = np.std(data.df['I_flux'].values)

        Delta_F_ = std_F*self.Delta_F + median_F
        F_base_ = std_F*self.Delta_F + median_F

        # Calculate source flux and blend flux
        FS = Delta_F_/(self.peak_mag() - 1)
        FB = (F_base_ - FS)/FS

        # Convert fluxes to magnitudes
        mu_m, sig_m = data.fluxes_to_magnitudes(np.array([FS, FB]), 
            np.array([0., 0.]))
        mag_source, mag_blend = mu_m

        return mag_source, mag_blend

    def peak_mag(self):
        """Returns PSPL magnification at u=u0."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

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