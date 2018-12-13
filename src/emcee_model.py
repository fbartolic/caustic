import numpy as np
import emcee

class PointSourcePointLens_emcee(object):
    """Class defining a PSPL emcee model."""
    def __init__(self, t, F, sigF, *args, **kwargs):
        self.t = t
        self.F =  F
        self.sigF = sigF

    def forward_model(self, pars, t):
        DeltaF, Fb, t0, teff, tE, u_K = pars

        t0 = (t[-1] - t[0])*t0 + t[0]
        
        u0 = teff/tE

        u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
        A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))

        return DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

    def log_likelihood(self, pars):
        DeltaF, Fb, t0, teff, tE, u_K = pars

        if u_K < 0.:
            K = 1.
        else:
            K = 1. - np.log(1. - u_K)

        dF = self.F - self.forward_model(pars, self.t)

        sigF = K*self.sigF # Rescale errorbars

        return np.sum(-0.5*dF**2/sigF**2)

    def log_prior(self, pars):
        lnL = 0

        DeltaF, Fb, t0, teff, tE, u_K = pars
        
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
        
        # u_K prior
        if u_K < -1. or u_K > 1.:
            return -np.inf

        return lnL

    def log_posterior(self, pars):
        lnp = self.log_prior(pars)
        if np.isinf(lnp): # short-circuit if the prior is infinite (don't bother computing likelihood)
            return lnp

        lnL = self.log_likelihood(pars)
        lnprob = lnp + lnL

        if np.isnan(lnprob):
            return -np.inf

        return lnprob

    def sample(self, nsteps, nwalkers):
        t0_guess_idx = (np.abs(self.F - np.max(self.F))).argmin()

        # Starting position of the chains
        initial = [np.max(self.F), 0., 
            (self.t[t0_guess_idx] - self.t[0])/(self.t[-1] - self.t[0]), 
            0.1, 10., 0.]

        ndim, nwalkers = len(initial), nwalkers
        p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()

        print("Running production...")
        sampler.run_mcmc(p0, nsteps)

        return sampler