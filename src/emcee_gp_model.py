import numpy as np
from matplotlib import pyplot as plt
import emcee
import sys
sys.path.append('codebase')
from plotting_utils import plot_data, plot_emcee_traceplots
from data_preprocessing_ogle import process_data
import celerite
from celerite.modeling import Model # an abstract class implementing the 
# skeleton of the celerite modeling protocol
from celerite import terms
from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

def solve_for_invgamma_params(params, t_min, t_max):
    def inverse_gamma_cdf(x, alpha, beta):
        return invgamma.cdf(x, alpha, scale=beta)

    alpha, beta = params

    return (inverse_gamma_cdf(2*t_min, alpha, beta) - \
    0.01, inverse_gamma_cdf(t_max, alpha, beta) - 0.99)


class CustomCeleriteModel(Model):
    """Celerite requires the specification of a custom class implementing
    a model. This is subclass of the abstract Model class from Celerite."""
    parameter_names = ("DeltaF", "Fb", "t0", "teff", "tE")

    def get_value(self, t):
        u0 = self.teff/self.tE
        u = np.sqrt(u0**2 + ((t - self.t0)/self.tE)**2)
            
        A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))  
        
        return self.DeltaF*(A(u) - 1)/(A(u0) - 1) + self.Fb

class PointSourcePointLensGP_emcee(object):
    """Class defining a PSPL emcee model using  a Celerite GP for the noise
        model."""
    def __init__(self, t, F, sigF, *args, **kwargs):
        self.t = t
        self.F =  F
        self.sigF = sigF

        # Set up mean model
        t0_guess_idx = (np.abs(F - np.max(F))).argmin()

        mean_model = CustomCeleriteModel(0., 0., self.t[t0_guess_idx], 0.1, 10.)

        # Set up the GP model
        term1 = terms.Matern32Term(log_sigma=np.log(2.), log_rho=np.log(10))
        kernel = term1

        gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
        gp.compute(self.t, self.sigF)

        self.gp = gp

        print("Initial log-likelihood: {0}".format(gp.log_likelihood(F)))
        print("Parameter names", gp.parameter_names)

        # Compute parameters for the prior on GP hyperparameters
        a, b =  fsolve(solve_for_invgamma_params, (0.1, 0.1), 
            (np.median(np.diff(self.t)), t[-1] - t[0]))
        #print("Median delta_t: ", np.diff(self.t))
        #print("Max t", t[-1] - t[0])
        self.rho_invgamma_params = [a, b]

    def log_prior(self, pars):

        lnL = 0 

        ln_sigma, ln_rho, DeltaF, Fb, t0, teff, tE, u_K = pars

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
        if u_K < -1 or u_K > 1:
            return -np.inf

        # DeltaF prior
        if DeltaF < 0:
            return -np.inf
        lnL += -(DeltaF - np.max(self.F))**2/1.**2
        
        # Fb prior
        lnL += -Fb**2/0.1**2
        
        # t0 prior 
        if t0 < np.min(self.t) or t0 > np.max(self.t):
            return -np.inf
        
        # (lnteff, lntE) joint prior
        sig_u0 = 1.
        sig_tE = 365.
        lnL += -np.log(tE) - (teff/tE)**2/sig_u0**2\
            - tE**2/sig_tE**2
        
        return lnL

    def log_posterior(self, pars):    
        ln_sigma, ln_rho, DeltaF, Fb, t0, teff, tE, u_K = pars

        if u_K < 0:
            K = 1.
        else:
            K = 1 - np.log(1 - u_K)

        self.gp.compute(self.t, K*self.sigF)
        self.gp.set_parameter_vector((ln_sigma,ln_rho,DeltaF,Fb,t0,teff,tE))
        
        lp = self.log_prior(pars)
        
        if not np.isfinite(lp):
            return -np.inf
        return self.gp.log_likelihood(self.F) + lp 

    def sample(self, nsteps, nwalkers):
        initial_pars_gp = self.gp.get_parameter_vector()
        initial_pars = np.append(initial_pars_gp, (0,))

        ndim, nwalkers = len(initial_pars), nwalkers
        p0 = initial_pars + 1e-8 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()

        print("Running production...")
        sampler.run_mcmc(p0, nsteps);
        
        return sampler, self.gp

import os 
events = [] # event names
lightcurves = [] # data for each event

i = 0
n_events = 100
for entry in os.scandir('../../../data/OGLE_ews/2017/'):
    if entry.is_dir() and (i < n_events):
        events.append(entry.name)
        photometry = np.genfromtxt(entry.path + '/phot.dat', usecols=(0,1,2))
        lightcurves.append(photometry)
        i = i + 1