import numpy as np
import emcee
import celerite
from celerite.modeling import Model # an abstract class implementing the 
# skeleton of the celerite modeling protocol
from celerite import terms

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

        mean_model = CustomCeleriteModel(0., 0., t[t0_guess_idx], 0.1, 10.)

        # Set up the GP model
        term1 = terms.SHOTerm(log_S0=0., log_Q=0., log_omega0=0.)
        term2 = terms.SHOTerm(log_S0=np.log(1), log_Q=np.log(1/np.sqrt(2)), log_omega0=0.)
        term2.freeze_parameter('log_S0')
        term2.freeze_parameter('log_Q')
        kernel = term1 * term2

        gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
        gp.compute(t, sigF)

        self.gp = gp

        print("Initial log-likelihood: {0}".format(gp.log_likelihood(F)))
        print("Parameter names", gp.parameter_names)

    def log_prior(self, pars):
        lnL = 0
        ln_S0, ln_Q, ln_omega1, ln_omega2, DeltaF, Fb, t0, teff, tE = pars

        # Priors for the GP hyperparameters
        if ln_S0 < -15. or ln_S0 > 5.:
            return -np.inf
        if ln_Q < -10. or ln_Q > 15:
            return -np.inf
        if ln_omega1 < -10. or ln_omega2 > 10:
            return -np.inf
        if ln_omega2 < -5. or ln_omega2 > 5:
            return -np.inf

        # DeltaF prior
        if DeltaF < 0:
            return -np.inf
        lnL += -(DeltaF - np.max(self.F))**2/1.**2
        
        # Fb prior
        lnL += -Fb**2/0.1**2
        
        # t0 prior 
        if t0 < 0. or t0 > 1.:
            return -np.inf
        
        # (lnteff, lntE) joint prior
        sig_u0 = 1.
        sig_tE = 365.
        lnL += -np.log(tE) - (teff/tE)**2/sig_u0**2\
            - tE**2/sig_tE**2
        
        return lnL

    def log_posterior(self, pars):    
        self.gp.set_parameter_vector(pars)
        
        lp = self.log_prior(pars)
        
        if not np.isfinite(lp):
            return -np.inf
        return self.gp.log_likelihood(self.F) + lp 

    def sample(self, nsteps, nwalkers):
        initial_gp = self.gp.get_parameter_vector()
        ndim, nwalkers = len(initial_gp), nwalkers
        p0_gp = initial_gp + 1e-8 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0_gp, 1000)
        sampler.reset()

        print("Running production...")
        sampler.run_mcmc(p0_gp, nsteps);
        
        return sampler

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
 
gp_model = PointSourcePointLensGP_emcee(lightcurves[0][:, 0], 
    lightcurves[0][:, 1], lightcurves[0][:, 2])

sampler = gp_model.sample(5000, 50)
print(sampler.acceptance_fraction)


labels = ['$\ln S_0$', '$\ln Q$', '$\ln\omega_1$', '$\ln\omega_2$',\
    '$\Delta F$', '$F_b$', '$t_0$', '$t_{eff}$', '$t_E$']

import sys
sys.path.append('codebase')
from plotting_utils import plot_data, plot_emcee_traceplots
fig, ax = plot_emcee_traceplots(sampler,
        labels=labels, acceptance_fraction=0.)
plt.show()