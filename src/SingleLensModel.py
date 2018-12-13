import numpy as np
import sys
sys.path.append("../../../exoplanet")
sys.path.append("../codebase")

import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import exoplanet as xo

print("PyMC3 version {0}".format(pm.__version__))

def initialize_model(t, F, sigF):
    model = pm.Model()


    with model:    
        # Priors
        ## log density of the joint prior for (tE, teff)
        def joint_density(value):
            teff = T.cast(value[0], 'float64')
            tE = T.cast(value[1], 'float64')
            sig_tE = T.cast(365., 'float64')
            sig_u0 = T.cast(1., 'float64')
            return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive

        # Noise model parameters
        ln_K = pm.Normal('ln_K', mu=0., sd=1.5, testval=0.)
        K = T.exp(ln_K) + 1.
        #u_K = pm.Uniform('u_K', -1., 1.)
        #K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))

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
        
        Y_obs = pm.Normal('Y_obs', mu=mean_function(t), sd=K*sigF, 
            observed=F)

        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point))

    return model