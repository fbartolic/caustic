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
from exoplanet.utils import get_samples_from_trace
from exoplanet.utils import eval_in_model

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve


class PointSourcePointLensClassic(pm.Model):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, use_joint_prior=True, name='', model=None):
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super(PointSourcePointLensClassic, self).__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.FS = BoundedNormal('FS', mu=np.max(self.F), sd=5., testval=3.)
        self.FB = pm.Normal('Fb', mu=0., sd=3, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])
        self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
        self.tE = BoundedNormal('tE', mu=0., sd=600., testval=20.)

        # Noise model parameters
        self.K = BoundedNormal1('K', mu=1., sd=2., testval=1.5)

        Y_obs = pm.Normal('Y_obs', mu=self.mean_function(), sd=self.K*self.sigF, 
            observed=self.F, shape=len(self.F))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.FS*A(u)  + self.FB

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

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(self.F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])
        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=600., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.joint_density, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Save source flux and blend parameters
        self.FS = pm.Deterministic("FS", self.DeltaF/(self.peak_mag() - 1))
        self.g = pm.Deterministic("g", (self.Fb - self.FS)/self.FS)

        # Noise model parameters
        self.K = BoundedNormal1('K', mu=1., sd=2., testval=1.5)

        Y_obs = pm.Normal('Y_obs', mu=self.mean_function(), sd=self.K*self.sigF, 
            observed=self.F, shape=len(self.F))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def peak_mag(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

    def joint_density(self, value):
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]

class PointSourcePointLensStudentT(pm.Model):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, use_joint_prior=True, name='', model=None):
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super(PointSourcePointLensStudentT, self).__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(self.F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])
        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=600., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.joint_density, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Save source flux and blend parameters
        self.FS = pm.Deterministic("FS", self.DeltaF/(self.peak_mag() - 1))
        self.g = pm.Deterministic("g", (self.Fb - self.FS)/self.FS)

        # Noise model parameters
        self.K = BoundedNormal1('K', mu=1., sd=2., testval=1.5)
        self.nu = pm.Gamma('nu', 2., 0.1)

        Y_obs = pm.StudentT('Y_obs', nu=self.nu, mu=self.mean_function(), 
            sd=self.K*self.sigF, observed=self.F, shape=len(self.F))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def peak_mag(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)


    def joint_density(self, value):
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]

class PointSourcePointLensRescaling(pm.Model):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, use_joint_prior=True, name='', model=None):
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super(PointSourcePointLensRescaling, self).__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(self.F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])
        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=600., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.joint_density, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Save source flux and blend parameters
        self.FS = pm.Deterministic("FS", self.DeltaF/(self.peak_mag() - 1))
        self.g = pm.Deterministic("g", (self.Fb - self.FS)/self.FS)


        # Noise model parameters
        self.alpha = BoundedNormal1('alpha', mu=1.05, sd=3, testval=1.5)
        self.K = pm.DensityDist('K', self.prior_for_K, shape=len(self.F),
            testval = np.random.multivariate_normal(2*np.ones(len(self.F), 
                0.1*np.ones(len(self.F)))))

        Y_obs = pm.Normal('Y_obs', mu=self.mean_function(), sd=self.K*self.sigF, 
            observed=self.F, shape=len(self.F))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def peak_mag(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

    def joint_density(self, value):
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]

    def prior_for_K(self, value):
        return T.log(self.alpha - 1) - self.alpha*T.log(value)

class PointSourcePointLensMatern32(pm.Model):
    def __init__(self, data, use_joint_prior=True, name='', model=None):
        super(PointSourcePointLensMatern32, self).__init__(name, model)

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        # Compute parameters for the prior on GP hyperparameters
        invgamma_a, invgamma_b =  fsolve(self.solve_for_invgamma_params, (0.1, 0.1), 
        (np.min(np.diff(self.t)), self.t[-1] - self.t[0]))

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(self.F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])

        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=400., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.joint_density, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Save source flux and blend parameters
        self.FS = pm.Deterministic("FS", self.DeltaF/(self.peak_mag() - 1))
        self.g = pm.Deterministic("g", (self.Fb - self.FS)/self.FS)

        # Noise model parameters
        self.sigma = BoundedNormal('sigma', mu=0., sd=3., testval=0.5)
        self.rho = pm.InverseGamma('rho', alpha=invgamma_a, beta=invgamma_b, 
            testval=3.)
        self.K = BoundedNormal1('K', mu=1.001, sd=2., testval=1.5)

        # Calculate likelihood
        kernel = terms.Matern32Term(sigma=self.sigma, rho=self.rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        self.gp = GP(kernel, self.t, (self.K*self.sigF)**2, J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", self.gp.log_likelihood(self.F - self.mean_function()))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def peak_mag(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

    def joint_density(self, value):
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]

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

class PointSourcePointLensSHO(pm.Model):
    def __init__(self, data, use_joint_prior=True, name='', model=None):
        super(PointSourcePointLensSHO, self).__init__(name, model)

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        # Compute parameters for the prior on GP hyperparameters
        invgamma_a, invgamma_b =  fsolve(self.solve_for_invgamma_params, (0.1, 0.1), 
            (np.min(np.diff(self.t)), self.t[-1] - self.t[0]))

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(self.F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])

        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=400., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.joint_density, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Save source flux and blend parameters
        self.FS = pm.Deterministic("FS", self.DeltaF/(self.peak_mag() - 1))
        self.g = pm.Deterministic("g", (self.Fb - self.FS)/self.FS)

        # Noise model parameters
        self.S0= BoundedNormal('S0', mu=0., sd=3., testval=0.5)
        self.rho = pm.InverseGamma('rho', alpha=invgamma_a, beta=invgamma_b, 
            testval=3.)
        self.Q = BoundedNormal('Q', mu=0., sd=100., testval=1/np.sqrt(2))
        self.K = BoundedNormal1('K', mu=1.001, sd=2., testval=1.5)

        # Calculate likelihood
        term1 = terms.SHOTerm(S0=self.S0, Q=self.Q, w0=2*np.pi/self.rho)
        kernel = term1
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        self.gp = GP(kernel, self.t, (self.K*self.sigF)**2, J=2) # J=2 for SHO kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", self.gp.log_likelihood(self.F - self.mean_function()))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def peak_mag(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

    def joint_density(self, value):
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]
    
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

class PointSourcePointLensSHOProduct(pm.Model):
    def __init__(self, data, n_terms=1, use_joint_prior=True, name='', model=None):
        super(PointSourcePointLensSHOProduct, self).__init__(name, model)
        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        # Compute parameters for the prior on GP hyperparameters
        self.invgamma_a, self.invgamma_b =  fsolve(self.solve_for_invgamma_params,
            (0.1, 0.1), (np.min(np.diff(self.t)), self.t[-1] - self.t[0]))
        print(self.invgamma_a)
        print(self.invgamma_b)

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Microlensing model parameters
        self.DeltaF = BoundedNormal('DeltaF', mu=np.max(self.F), sd=1., testval=3.)
        self.Fb = pm.Normal('Fb', mu=0., sd=0.1, testval=0.)
        # Posterior is multi-modal in t0 and it's critical that the it is 
        # initialized near the true value
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin() 
        self.t0 = pm.Uniform('t0', self.t[0], self.t[-1], 
            testval=self.t[t0_guess_idx])

        if (use_joint_prior==False):
            self.u0 = BoundedNormal('u0', mu=0., sd=1., testval=0.5)
            self.tE = BoundedNormal('tE', mu=0., sd=400., testval=20.)
        else:
            self.ln_teff_ln_tE = pm.DensityDist('ln_teff_ln_tE', 
                self.joint_density, shape=2, 
                testval = [np.log(10), np.log(20.)]) # p(ln_teff,ln_tE)

            # Deterministic transformations
            self.tE = pm.Deterministic("tE", T.exp(self.ln_teff_ln_tE[1]))
            self.u0 = pm.Deterministic("u0", 
                T.exp(self.ln_teff_ln_tE[0])/self.tE) 

        # Save source flux and blend parameters
        self.FS = pm.Deterministic("FS", self.DeltaF/(self.peak_mag() - 1))
        self.g = pm.Deterministic("g", (self.Fb - self.FS)/self.FS)


        # Noise model parameters
        self.S0= BoundedNormal('S0', mu=0., sd=3., testval=0.5)
        self.ln_omega0 = pm.DensityDist('ln_omega0', self.prior_for_ln_omega,
            testval=3)
        self.Q = BoundedNormal('Q', mu=0., sd=100., testval=1/np.sqrt(2))
        self.K = BoundedNormal1('K', mu=1.001, sd=2., testval=1.5)

        # Calculate likelihood
        term1 = terms.SHOTerm(S0=self.S0, Q=self.Q, log_w0=self.ln_omega0)
        kernel = term1
        
        if (n_terms >= 2):
            self.ln_omega1 = pm.DensityDist('ln_omega1', self.prior_for_ln_omega,
                testval=3)
            self.S0_1= BoundedNormal('S0_1', mu=0., sd=3., testval=0.5)
            self.Q_1 = BoundedNormal('Q_1', mu=0., sd=100., testval=1/np.sqrt(2))
            term2 = terms.SHOTerm(S0=self.S0_1, Q=self.Q_1, log_w0=self.ln_omega1)
            kernel *= term2

        if (n_terms >= 3):
            self.ln_omega2 = pm.DensityDist('ln_omega2', self.prior_for_ln_omega,
                testval=3)
            term3 = terms.SHOTerm(S0=1, Q=1/np.sqrt(2), log_w0=self.ln_omega2)
            kernel *= term3

        if (n_terms >= 4):
            self.ln_omega3 = pm.DensityDist('ln_omega3', self.prior_for_ln_omega,
                testval=3)
            term4 = terms.SHOTerm(S0=1, Q=1/np.sqrt(2), log_w0=self.ln_omega3)
            kernel *= term4
       
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        self.gp = GP(kernel, self.t, (self.K*self.sigF)**2) # J=2 for SHO kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", self.gp.log_likelihood(self.F - self.mean_function()))

    def mean_function(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return self.DeltaF*(A(u) - 1)/(A(self.u0) - 1) + self.Fb

    def peak_mag(self):
        """PSPL model"""
        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return A(self.u0)

    def joint_density(self, value):
        teff = T.cast(T.exp(value[0]), 'float64')
        tE = T.cast(T.exp(value[1]), 'float64')
        sig_tE = T.cast(365., 'float64') # p(tE)~N(0, 600)
        sig_u0 = T.cast(1., 'float64') # p(u0)~N(0, 1)

        return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2 +\
            value[0] + value[1]
    
    def prior_for_ln_omega(self, value):
        """
        Custom prior for the ln_omega0 parameter in the SHO kernel (the resonant
        frequency. The prior is derived from a an Inverse Gamma prior on
        P0 = 2pi/omega0.
        """
        a = self.invgamma_a
        b = self.invgamma_b
        omega0 = T.cast(T.exp(value), 'float64')
        P0 = 2*np.pi/omega0
        return T.log(2*np.pi) - T.log(omega0) + a*T.log(b) - T.log(gamma(a)) -\
            (a + 1)*T.log(P0) - b/P0
        

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


class ZeroMeanMatern32(pm.Model):
    def __init__(self, data, name='', model=None):
        super(ZeroMeanMatern32, self).__init__(name, model)

        # Pre process the data
        data.convert_data_to_fluxes()
        df = data.get_standardized_data()

        # Data
        self.t = df['HJD - 2450000'].values
        self.F = df['I_flux'].values
        self.sigF = df['I_flux_err'].values

        # Custom prior distributions 
        # Compute parameters for the prior on GP hyperparameters
        invgamma_a, invgamma_b =  fsolve(self.solve_for_invgamma_params, (0.1, 0.1), 
        (np.min(np.diff(self.t)), self.t[-1] - self.t[0]))


        BoundedNormal = pm.Bound(pm.Normal, lower=0.) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Noise model parameters
        self.sigma = BoundedNormal('sigma', mu=0., sd=3., testval=0.5)
        self.rho = pm.InverseGamma('rho', alpha=invgamma_a, beta=invgamma_b, 
            testval=3.)
        self.K = BoundedNormal1('K', mu=1.001, sd=2., testval=1.5)

        # Calculate likelihood
        kernel = terms.Matern32Term(sigma=self.sigma, rho=self.rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        self.gp = GP(kernel, self.t, (self.K*self.sigF)**2, J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", self.gp.log_likelihood(self.F))

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

class ZeroMeanMatern32StudentT(pm.Model):
    def __init__(self, data, name='', model=None):
        super(ZeroMeanMatern32StudentT, self).__init__(name, model)

        # Data
        self.t = data.df['HJD - 2450000'].values
        self.F = data.df['I_flux'].values
        self.sigF = data.df['I_flux_err'].values

        # Custom prior distributions 
        # Compute parameters for the prior on GP hyperparameters
        invgamma_a, invgamma_b =  fsolve(self.solve_for_invgamma_params, (0.1, 0.1), 
        (np.min(np.diff(self.t)), self.t[-1] - self.t[0]))

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Noise model parameters
        self.sigma = BoundedNormal('sigma', mu=0., sd=3., testval=0.5)
        self.rho = pm.InverseGamma('rho', alpha=invgamma_a, beta=invgamma_b, 
            testval=3.)
        self.K = BoundedNormal1('K', mu=1.001, sd=2., testval=1.5)

        # Prior for the latent function f which is modelled by a GP
        mu_f = self.F
        cov_f  = np.diag(10**2*np.ones(len(self.F)))
        self.f = pm.MvNormal('f', mu=mu_f, cov=cov_f, shape=(len(mu_f), ),
            testval=0.1*np.ones(len(mu_f)))

        # Calculate likelihood
        kernel = terms.Matern32Term(sigma=self.sigma, rho=self.rho)
        self.gp = GP(kernel, self.t, 1.e-8*np.ones(len(self.t)),  J=2) 

        # Add a custom "potential" (log probability function) with the GP likelihood
        lnL1 = self.gp.log_likelihood(self.f) 

        # Calculate the student-t likelihood
        lnL2 = self.mv_studentt_log_density(4.)
        lnL = lnL1 + lnL2

        pm.Potential("gp", lnL)

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

    def mv_studentt_log_density(self, nu):
        r = T._shared(self.F) - self.f
        lnL = np.log(gamma(0.5*(nu + 1))) - np.log(gamma(0.5*nu)) - 0.5*np.log(nu) 
        lnL = T.cast(lnL, 'float64')
        lnL += - 0.5*T.log(T.prod(self.K*self.sigF))
        lnL += -(0.5*(nu + 1))*T.log(1 + T.sum(r**2/(self.K*self.sigF)**2)/nu)
        return lnL