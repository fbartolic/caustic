import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import emcee
import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import seaborn as sns
import os
import sys
sys.path.append('codebase')
from data_preprocessing_ogle import process_data
from plotting_utils import plot_data, plot_emcee_traceplots

from emcee_model import PointSourcePointLens_emcee
from emcee_gp_model import PointSourcePointLensGP_emcee

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16

events = [] # event names
lightcurves = [] # data for each event
 
i = 0
n_events = 20
for entry in os.scandir('../../../data/OGLE_ews/2017/'):
    if entry.is_dir() and (i < n_events):
        events.append(entry.name)
        photometry = np.genfromtxt(entry.path + '/phot.dat', usecols=(0,1,2))
        lightcurves.append(photometry)
        i = i + 1
        
print("Loaded events:", events)

def fit_pymc3_model(t, F, sigF):
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
        u_K = pm.Uniform('u_K', -1., 1.)

        K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))

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
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=K*sigF, observed=F)

    t0_guess_idx = (np.abs(F - np.max(F))).argmin()

    # Initialization of the chain
    start = {'DeltaF':np.max(F), 'Fb':0.,
        't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}
    
    # Fit model with NUTS
    with model:
        trace = pm.sample(2000, tune=2000, nuts_kwargs=dict(target_accept=.95),
            start=start)
    
    return trace 
        
# Iterate over events, fit them and save the plots
for event_index, lightcurve in enumerate(lightcurves):
    # Pre process data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
         lightcurve[:, 2], standardize=True)

    # Save processed data
    if not os.path.exists('output/' + events[event_index]):
        os.makedirs('output/' + events[event_index])
    
    data = np.stack((t, F, sigF), axis=1)
    np.save('output/' + events[event_index] + '/data.npy', 
        data)

    # Fit pymc3 model
    trace = fit_pymc3_model(t, F, sigF)

    # Calculate number of divergent samples 
    divergent1 = trace['diverging']
    divperc1 = divergent1.nonzero()[0].size / len(trace)*100

    # Fit emcee model
    model_emcee = PointSourcePointLens_emcee(t, F, sigF)
    sampler_emcee = model_emcee.sample(5000., 50)

    # Fit emcee model with GP
    model_emcee_GP = PointSourcePointLensGP_emcee(t, F, sigF)
    sampler_emcee_GP, gp = model_emcee_GP.sample(5000., 50)

    # Save posterior samples
    samples_pymc3 = np.vstack([trace['DeltaF'],trace['Fb'],trace['t0'],
                         trace['teff_tE'][:, 0],trace['teff_tE'][:, 1],
                         trace['u_K']]).T
    samples_emcee = sampler_emcee.chain
    samples_emcee_GP = sampler_emcee_GP.chain

    np.save('output/' + events[event_index] + '/samples_pymc3.npy', 
        samples_pymc3)
    np.save('output/' + events[event_index] + '/samples_emcee.npy',
        samples_emcee)
    np.save('output/' + events[event_index] + '/samples_emcee_GP.npy',
        samples_emcee_GP)
    np.save('output/' + events[event_index] + '/samples_emcee_acc_frac.npy',
        sampler_emcee.acceptance_fraction)
    np.save('output/' + events[event_index] + '/samples_emcee_GP_acc_frac.npy',
        sampler_emcee_GP.acceptance_fraction)

    # Save pymc3 traceplots
    fig, ax = plt.subplots(5, 2 ,figsize=(10,10))
    _ = pm.traceplot(trace, ax=ax)
    plt.savefig('output/' + events[event_index] + '/traceplots.png')

    # Save a log file
    with open('output/' + events[event_index] + "/log.txt", "w") as text_file:
        print(f"Percentage of divergent points: {divperc1} \n", file=text_file)
        print(f"Median spacing between data points: \
            {np.median(np.diff(t))} \n", file=text_file)