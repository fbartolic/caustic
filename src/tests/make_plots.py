import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import pymc3 as pm
import os
import time

import sys
sys.path.append("../theano_ops")
sys.path.append("../codebase")
from data_preprocessing_ogle import process_data
from plotting_utils import *

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
        term1 = celerite_terms.Matern32Term(log_sigma=np.log(2.), 
            log_rho=np.log(10))
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

limits = [1700, 1900, 1900, 1700, 2500]

events = [] # event names
lightcurves = [] # data for each event
 
i = 0
n_events = 5
data_path = '/home/star/fb90/data/OGLE_ews/2017'
for entry in sorted(os.listdir(data_path)):
    if (i < n_events):
        events.append(entry)
        print(entry)
        photometry = np.genfromtxt(data_path + '/' + entry + '/phot.dat', usecols=(0,1,2))
        lightcurves.append(photometry)
        i = i + 1
        
print("Loaded events:", events)

def plot_posterior_samples(ax, samples):
    samples = samples.T

    for s in samples[np.random.randint(len(samples), 
            size=100)]:
        gp.set_parameter_vector(s[:-1])
        gp.compute(t, (1. + np.exp(s[-1]))*sigF)
        mu = gp.predict(F, t_, return_cov=False)
        ax.plot(t_, mu, color='C1', alpha=0.3)
    ax.grid(True)

# Make pymc3 plots
for event_index, lightcurve in enumerate(lightcurves):
    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

    t = t[:limits[event_index]]
    F = F[:limits[event_index]]
    sigF = sigF[:limits[event_index]]

    # Load samples
    samples_pymc3 = np.load('output/' + events[event_index] + '/samples_pymc3.npy')
    quantiles_pymc3_GP = np.percentile(samples_pymc3,
            [16, 50, 84], axis=1)

    model_emcee_GP = GP_emcee(t, F, sigF)
    gp = model_emcee_GP.gp

    median_GP_params_pymc3 = quantiles_pymc3_GP[1, :-1]

    t_ = np.linspace(t[0], t[-1], 5000)

    # Residuals
    gp.set_parameter_vector(median_GP_params_pymc3)
    gp.compute(t, (1. + np.exp(median_GP_params_pymc3[-1]))*sigF)
    mu_pymc3 = gp.predict(F, t_, return_cov=False)
    residuals_GP_pymc3 = F - gp.predict(F, t, return_cov=False)

    # Plot posterior samples pymc3
    plt.clf()
    fig, ax = plt.subplots(figsize=(25, 6))
    plot_data(ax, t, F, sigF) # Plot data
    ax.set_xlim(t[0], t[-1])
    plot_posterior_samples(ax, samples_pymc3)
    plt.savefig('output/' + events[event_index] + '/model_pymc3_GP.png')    

    # Plot corner plot pymc3
    plt.clf()
    fig = corner.corner(samples_pymc3.T)
    fig.constrained_layout = True
    plt.savefig('output/' + events[event_index] + '/corner_pymc3.png')

# Make emcee plots
for event_index, lightcurve in enumerate(lightcurves):
    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

    t = t[:limits[event_index]]
    F = F[:limits[event_index]]
    sigF = sigF[:limits[event_index]]

    # Load samples
    samples_emcee = np.load('output/' + events[event_index] + '/samples_emcee.npy')    
    
    # Plot emcee traceplots
    labels_GP = ['ln_sigma', 'ln_rho']
    fig1, ax1 = plot_emcee_traceplots(samples_emcee,
        labels_GP, np.ones(len(samples_emcee)), acceptance_fraction_cutoff=0.05)
    plt.savefig('output/' + events[event_index] + '/emcee_traceplots.png')    

    # Reshape emcee samples to standard form
    samples_emcee = samples_emcee.reshape(-1, 2).T
    quantiles_emcee_GP = np.percentile(samples_emcee,
            [16, 50, 84], axis=1)
    model_emcee_GP = GP_emcee(t, F, 1.*sigF)
    gp = model_emcee_GP.gp

    median_GP_params_emcee = quantiles_emcee_GP[1, :]

    t_ = np.linspace(t[0], t[-1], 5000)

    # Residuals
    gp.set_parameter_vector(median_GP_params_emcee)
    gp.compute(t, 1.*sigF)
    mu_emcee = gp.predict(F, t_, return_cov=False)
    residuals_GP_emcee = F - gp.predict(F, t, return_cov=False)

    # Plot model
    plt.clf()
    fig, ax = plt.subplots(figsize=(25, 6))
    plot_data(ax, t, F, sigF) # Plot data
    ax.set_xlim(t[0], t[-1])
    plot_posterior_samples(ax, samples_emcee)
    plt.savefig('output/' + events[event_index] + '/model_emcee_GP.png')    

    # Plot corner plot emcee
    plt.clf()
    fig = corner.corner(samples_emcee.reshape(-1, 2))
    fig.constrained_layout = True
    plt.savefig('output/' + events[event_index] + '/corner_emcee.png')
