import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import pymc3 as pm
import theano
import theano.tensor as T
import seaborn as sns
import os
import sys
sys.path.append('codebase')
from data_preprocessing_ogle import process_data
from plotting_utils import plot_data

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16

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
        
print("Loaded events:", events)

def fit_hierarchical_model(t, F, sigF):
    model = pm.Model()

    with model:    
        # log density of the joint prior for (tE, teff)
        def joint_density(value):
            teff = T.cast(value[0], 'float64')
            tE = T.cast(value[1], 'float64')
            sig_tE = T.cast(365., 'float64')
            sig_u0 = T.cast(1., 'float64')
            return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2

        # Priors for unknown model parameters
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1.)
        Fb = pm.Normal('Fb', mu=0., sd=0.1)
        t0 = pm.Uniform('t0', 0, 1.)
        teff_tE = pm.DensityDist('teff_tE', joint_density, shape=2, 
            testval = [0.1, 10.])
#         K = BoundedNormal2('K', mu=1., sd=1.)
        
        # Transform t0 to sensible units
        t0 = (t[-1] - t[0])*t0 + t[0]

        teff = teff_tE[0]
        tE = teff_tE[1]
        u0 = teff/tE

        # Calculate magnification
        u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
        A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))

        mu = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

        # Likelihood 
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigF, observed=F)

    t0_guess_idx = (np.abs(F - np.max(F))).argmin()

    # Initialization of the chain
    start = {'DeltaF':np.max(F), 'Fb':0.,
        't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}
    
    # Fit model with NUTS
    with model:
        trace = pm.sample(5000, tune=2000, nuts_kwargs=dict(target_accept=.95),
            start=start)
    
    return trace 
        
# Iterate over events, fit them and save the plots
for event_index, lightcurve in enumerate(lightcurves):
    # Pre process data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
         lightcurve[:, 2], standardize=True)

    # Fit pymc3 model
    trace = fit_pymc3_model(t, F, sigF)
    # Calculate number of divergent samples 
    divergent1 = trace['diverging']
    divperc1 = divergent1.nonzero()[0].size / len(trace)*100
    
    
    # Plot traceplots
    fig, ax = plt.subplots(4, 2 ,figsize=(10,10))
    plt.title('Percentage of divergent samples %.1f' % divperc1)
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig('output/' + events[event_index] + '/traceplots.png')

    plt.clf()
    labels = ['$\Delta F$', '$F_b$', '$t_0$', '$t_{eff}$', '$t_E$']
    fig, ax = plot_emcee_traceplots(sampler_emcee,
        labels=labels, acceptance_fraction=0.05)

    # Plot model
    quantiles_pymc3 = np.percentile(samples_pymc3, [16, 50, 84], axis=0)

    t_ = np.linspace(t[0], t[-1], 1000)

    plt.clf()
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_data(ax, t, F, sigF)
    ax.plot(t_, model_emcee.forward_model(quantiles_pymc3[1], t_), 
                 marker='', linestyle='-', color='C0', lw=2., label='NUTS')
    ax.legend(prop={'size': 12})
    plt.savefig('output/' + events[event_index] + '/model.png')    

    # Plot corner plot
    plt.clf()
    fig = corner.corner(samples_pymc3, labels=labels)
    fig.constrained_layout = True
    plt.savefig('output/' + events[event_index] + '/corner_pymc3.png')

    # Plot posterior for important parameters
    plt.clf()
    fig, ax = plt.subplots( figsize=(8,8))
    ax.hist(samples_pymc3[:, -2], bins=50, normed=True, color='C0', alpha=0.7)
    ax.set_xlabel(r'$t_E$')