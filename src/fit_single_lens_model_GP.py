import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
sys.path.append("../../exoplanet")
sys.path.append("codebase")

import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import exoplanet as xo
from exoplanet.gp import terms, GP

from data_preprocessing_ogle import process_data
from plotting_utils import *
from SingleLensModelMatern32 import initialize_model

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve


mpl.rc('text', usetex=False)

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

np.random.seed(42)

for event_index, lightcurve in enumerate(lightcurves):
    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

    # Fit pymc3 model
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
        return (inverse_gamma_cdf(x_min, alpha, beta) - \
        lower_mass, inverse_gamma_cdf(x_max, alpha, beta) - upper_mass)

    model = pm.Model()

    # Compute parameters for the prior on GP hyperparameters
    invgamma_a, invgamma_b =  fsolve(solve_for_invgamma_params, (0.1, 0.1), 
        (np.median(np.diff(t)), t[-1] - t[0]))

    with pm.Model() as model:    
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
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Noise model parameters
        #ln_sigma = pm.DensityDist('ln_sigma', ln_sigma_prior, testval=np.log(2.))
        sigma = BoundedNormal('sigma', mu=0., sd=3., testval=0.5)
        #ln_rho = pm.DensityDist('ln_rho', ln_rho_prior, testval=np.log(10.))
        rho = pm.InverseGamma('rho', alpha=invgamma_a, beta=invgamma_b, 
            testval=3.)
        #ln_K = pm.Normal('ln_K', mu=0., sd=1.5, testval=0.)
        #K = T.exp(ln_K) + 1.
        K = BoundedNormal('K', mu=1.001, sd=2., testval=1.5)
        #u_K = pm.Uniform('u_K', -1., 1.)
        #K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))
        ln_rho = pm.Deterministic('ln_rho', T.log(rho))
        ln_sigma = pm.Deterministic('ln_sigma', T.log(sigma))

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

        #mean = pm.Deterministic("flux", mean_function(t))
        
        kernel = terms.Matern32Term(sigma=sigma, rho=rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        gp = GP(kernel, t, (K*sigF)**2, J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the GP likelihood
        pm.Potential("gp", gp.log_likelihood(F - mean_function(t)))

        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point))


    # Fit MAP model
    ##with model:
    #    map_soln = pm.find_MAP(start=model.test_point)

    t0_guess_idx = (np.abs(F - np.max(F))).argmin()
    # Initialization of the chain
    start = {'DeltaF':np.max(F), 'Fb':0., 'K':1.5, 'sigma':0.5, 
    'rho':np.exp(2.), 'teff_tE':[35., 55.], 't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}
    
    # Sample model with NUTS
    sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

    with model:
        burnin = sampler.tune(tune=3000, start=start, 
            step_kwargs=dict(target_accept=0.95))
            
    with model:
        trace = sampler.sample(draws=2000)

    # Save stats summary to file
    if not os.path.exists('output/' + events[event_index]):
        os.makedirs('output/' + events[event_index])
    if not os.path.exists('output/' + events[event_index] + 
        '/PointSourcePointLensGP'):
        os.makedirs('output/' + events[event_index] + '/PointSourcePointLensGP')

    stats = pm.summary(trace)
    stats = stats.round(2)
    stats.to_csv('output/' + events[event_index] + '/PointSourcePointLensGP' +\
        '/summary.npy')

    # Save posterior samples
    samples_pymc3 = pm.trace_to_dataframe(trace, 
        varnames=["sigma", "rho", "K","DeltaF", "Fb", "t0", 
            "teff_tE"]).values.T

    np.save('output/' + events[event_index] + '/PointSourcePointLensGP'  +\
        '/samples_pymc3.npy', samples_pymc3)

    # Save traceplots
    fig, ax = plt.subplots(10, 2 ,figsize=(20, 30))
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig('output/' + events[event_index] + '/PointSourcePointLensGP' +\
         '/traceplots_celerite_pymc3.png')

    # Plot model in data space
    plt.clf()
    t_ = np.linspace(t[0], t[-1], 1000)

    # Generate 50 realizations of the prediction sampling randomly from the chain
    N_pred = 50
    pred_mu = np.empty((N_pred, len(t_)))
    pred_var = np.empty((N_pred, len(t_)))
    mean_function = np.empty((N_pred, len(t_)))

    with model:
        pred = gp.predict(t_, return_var=True)
        for i, sample in enumerate(xo.get_samples_from_trace(trace, size=N_pred)):
            p = t0**2
            u = T.sqrt(u0**2 + ((t_ - t0)/tE)**2)
            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
            mean_func = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

            pred_mu[i], pred_var[i] = xo.eval_in_model(pred, sample)
            mean_function[i] = xo.eval_in_model(mean_func, sample)

    # Plot the predictions
    fig, ax = plt.subplots(figsize=(25, 6))
    for i in range(len(pred_mu)):
        mu = pred_mu[i]
        sd = np.sqrt(pred_var[i])
        label = None if i else "prediction"
        #art = plt.fill_between(t_, mu+sd, mu-sd, color="C1", alpha=0.1)
        #art.set_edgecolor("none")
        ax.plot(t_, mean_function[i] - mu, color="C1", label=label, alpha=0.1)
        
    plot_data(ax, t, F, sigF) # Plot data
    plt.savefig('output/' + events[event_index] + '/PointSourcePointLensGP/' +\
        'model.png')
    

    print(len(samples_pymc3[:, 0]))

    # display the total number and percentage of divergent
    plt.clf()
    divergent = trace['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print('Percentage of Divergent %.1f' % divperc)

    pm.pairplot(trace,
            sub_varnames=['ln_sigma','ln_rho'],
            divergences=True,
            color='C3', figsize=(10, 5), kwargs_divergence={'color':'C2'})
    plt.savefig('output/' + events[event_index] + '/PointSourcePointLensGP' +\
         '/divergences.png')