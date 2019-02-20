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
    def __init__(self, data, use_joint_prior=True, name='', model=None):
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
        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=600., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.prior_ln_teff_ln_tE, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Transform (Delta_F,F_basease) to (m_S,m_blend) using deterministic 
        # mapping and save to trace
        m_source, m_blend = self.revert_flux_params_to_nonstandardized_format(
            data)
        self.mag_source = pm.Deterministic("m_source", m_source)
        self.mag_blend = pm.Deterministic("m_blend", m_blend)

        # Noise model parameters
        self.K = BoundedNormal1('K', mu=1., sd=2., testval=1.5)

        # Save log prior for each parameter, this is needed for hierarchical
        # modeling of multiple events using the importance resampling trick
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            BoundedNormal.dist(mu=np.max(self.F), sd=1.).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(mu=0., sd=0.1).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t[0], self.t[-1]).logp(self.t0))
        self.logp_ln_teff_ln_tE= pm.Deterministic('logp_ln', 
            self.prior_ln_teff_ln_tE(self.ln_teff_ln_tE))
        self.logp_K = pm.Deterministic('logp_K',
            BoundedNormal1.dist( mu=1., sd=2.).logp(self.K))
        self.log_posterior = pm.Deterministic("log_posterior", self.logpt)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

        # Define the likelihood function~
        Y_obs = pm.Normal('Y_obs', mu=self.mean_function(), sd=self.K*self.sigF, 
            observed=self.F, shape=len(self.F))

    def mean_function(self):
        """Return the mean function which goes into the likeliood."""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.Delta_F*(A(u) - 1)/(A(self.u0) - 1) + self.F_base

    def prior_ln_teff_ln_tE(self, value):
        """Returns the log of a custom joint prior p(ln_Delta_F, ln_F_base)."""
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]

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
    def __init__(self, data, use_joint_prior=True, name='', model=None):
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
        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=600., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.prior_ln_teff_ln_tE, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Noise model parameters
        self.K = BoundedNormal1('K', mu=1., sd=2., testval=1.5)

        # Save log prior for each parameter, this is needed for hierarchical
        # modeling of multiple events using the importance resampling trick
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t[0], self.t[-1]).logp(self.t0))
        self.logp_ln_teff_ln_tE= pm.Deterministic('logp_ln', 
            self.prior_ln_teff_ln_tE(self.ln_teff_ln_tE))
        self.logp_K = pm.Deterministic('logp_K',
            BoundedNormal1.dist( mu=1., sd=2.).logp(self.K))
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
        C_diag = T.pow(self.K*T._shared(self.sigF), 2.)
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

    def prior_ln_teff_ln_tE(self, value):
        """Returns the log of a custom joint prior p(ln_teff, ln_tE)."""
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]