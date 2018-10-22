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
#sys.path.append('codebase')
from codebase.data_preprocessing_ogle import process_data
from codebase.plotting_utils import plot_data, plot_emcee_traceplots
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
    0.001, inverse_gamma_cdf(t_max, alpha, beta) - 0.99)

class CustomCeleriteModel(Model):
    """Celerite requires the specification of a custom class implementing
    a model. This is subclass of the abstract Model class from Celerite."""
    parameter_names = ("DeltaF", "Fb", "t0", "teff", "tE")

    def get_value(self, t):
        u0 = self.teff/self.tE
        u = np.sqrt(u0**2 + ((t - self.t0)/self.tE)**2)
            
        A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))  
        
        return self.DeltaF*(A(u) - 1)/(A(u0) - 1) + self.Fb

    # This method is optional but it can be used to compute the gradient of the
    # cost function below.
       
    def compute_gradient(self, t):
        u0 = self.teff/self.tE
        u = np.sqrt(u0**2 + ((t - self.t0)/self.tE)**2)
        A_u = (u**2 + 2)/(u*np.sqrt(u**2 + 4))  
        A_u0 = (u0**2 + 2)/(u0*np.sqrt(u0**2 + 4))  
        dAdu = -8./(u**2*(u**2 + 4)**(3/2.))
        dAdu0 = -8./(u0**2*(u0**2 + 4)**(3/2.))

        # dF/dDeltaF
        dF_dDeltaF = (A_u - 1)/(A_u0 - 1) 

        # dF/dFb
        dF_dFb = np.ones_like(t)

        # dF/dt0
        dF_dt0 = self.DeltaF/(A_u0 - 1)*dAdu*\
            (-(t - self.t0)/(u*self.tE**2))

        # dF/dteff
        dF_dteff =self.DeltaF*((dAdu/(u*self.tE))/(A_u0 - 1) - (A_u - 1)/\
            (A_u0 - 1)**2*dAdu0/self.tE)

        # dF/dtE
        dF_dtE = self.DeltaF*((-dAdu*((t - self.t0)**2/(u*self.tE**3)))\
        /(A_u0 - 1) - (A_u - 1)/(A_u0 - 1)**2*dAdu0*(self.teff/self.tE**2))

        gradients = np.array([dF_dDeltaF, dF_dFb, dF_dt0, dF_dteff, dF_dtE])
        #print("PARAMETERS: ", [self.DeltaF, self.Fb, self.t0, self.teff, 
            #self.tE], "\n")
        #print("GRADIENTS: ", gradients)
        return gradients

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
        gp.compute(self.t, 1.4*self.sigF)

        self.gp = gp

    def log_likelihood(self, pars):    
        ln_sigma, ln_rho, DeltaF, Fb, t0, teff, tE = pars

        #if u_K < 0:
        #    K = 1.
        #else:
        #    K = 1 - np.log(1 - u_K)

        #print("ln_sigma \n", ln_sigma)
        #print("ln_rho \n", ln_rho)
        #print("t0\n",  t0)
        #print("teff\n", teff)
        #print("tE\n", tE)
        #print("u_K\n ", u_K)

        self.gp.compute(self.t, 1.4*self.sigF)
        self.gp.set_parameter_vector((ln_sigma,ln_rho,DeltaF,Fb,t0,teff,tE))

        return self.gp.log_likelihood(self.F) 

# Define a theano Op for a custom likelihood function
class LogLike(T.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [T.dvector] # expects a vector of parameter values when called
    otypes = [T.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, gp, F):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        t:
            The dependent variable (aka 't') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, gp, F)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values 
        theta, = inputs  # our parameters 
        return [g[0]*self.logpgrad(theta)]

class LogLikeGrad(T.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [T.dvector]
    otypes = [T.dvector]

    def __init__(self, loglike, gp, F):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.gp = gp

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # calculate gradients
        gradients = self.gp.grad_log_likelihood(y=F)[1]

        # Raise an error if any of the gradients is a NaN
        if(np.any(np.isnan(gradients))):
            raise ValueError("The gradients are NaNs")

        # Check if any of the gradients are zero and raise an exception
        if np.any(np.allclose(gradients, 0)):
            print("Shape of gradients:", np.shape(gradients), "\n")
            print("Values of parameters:", np.shape(gradients), "\n")
            print("Parameter names:", self.gp.get_parameter_names, "\n")
            print("Parameter values:", self.gp.get_parameter_vector, "\n")

        outputs[0][0] = gradients


events = [] # event names
lightcurves = [] # data for each event
 
i = 0
n_events = 1
for entry in os.scandir('/home/star/fb90/data/OGLE_ews/2017/'):
    if entry.is_dir() and (i < n_events):
        events.append(entry.name)
        photometry = np.genfromtxt(entry.path + '/phot.dat', usecols=(0,1,2))
        lightcurves.append(photometry)
        i = i + 1
        
print("Loaded events:", events)

def fit_pymc3_model(t, F, sigF, custom_likelihood):
    model = pm.Model()

    # Compute parameters for the prior on GP hyperparameters
    invgamma_a, invgamma_b =  fsolve(solve_for_invgamma_params, (0.1, 0.1), 
        (np.median(np.diff(t)), t[-1] - t[0]))

    with model:    
        # log density of the joint prior for (tE, teff)
        def joint_density(value):
            teff = T.cast(value[0], 'float64')
            tE = T.cast(value[1], 'float64')
            sig_tE = T.cast(365., 'float64')
            sig_u0 = T.cast(1., 'float64')
            return -T.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2

        # Priors for GP hyperparameters
        
        def ln_rho_prior(ln_rho):
            lnpdf_lninvgamma = lambda  x, a, b: np.log(x) + a*np.log(b) -\
                 (a + 1)*np.log(x) - b/x - np.log(gamma(a)) 

            res = lnpdf_lninvgamma(np.exp(ln_rho), invgamma_a, invgamma_b)
            return T.cast(res, 'float64')

        ln_rho = pm.DensityDist('ln_rho', ln_rho_prior, testval = 0.)

        def ln_sigma_prior(ln_sigma):
            sigma = np.exp(ln_sigma)
            res = np.log(sigma) - sigma**2/3.**2
            return T.cast(res, 'float64')

        ln_sigma = pm.DensityDist('ln_sigma', ln_sigma_prior, testval = 2.)

        # Priors for unknown model parameters
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1.)
        Fb = pm.Normal('Fb', mu=0., sd=0.1)
        t0 = pm.Uniform('t0', 0, 1.)
        teff_tE = pm.DensityDist('teff_tE', joint_density, shape=2, 
            testval = [0.1, 10.])
        # u_K = pm.Uniform('u_K', -1., 1.)

        # K = ifelse(u_K < 0., T.cast(1., 'float64'), 1. - T.log(1. - u_K))

        # Transform t0 to sensible units
        t0 = (t[-1] - t[0])*t0 + t[0]

        teff = teff_tE[0]
        tE = teff_tE[1]
        u0 = teff/tE

        # Convert all parameters to a tensor vector
        theta = T.as_tensor_variable([ln_sigma, ln_rho, DeltaF, Fb, t0, teff,
            tE])

        ## Calculate the likelihood by calling a Theano Op 
        # use a DensityDist (use a lamdba function to "call" the Op)
        pm.DensityDist('likelihood', lambda v: custom_likelihood(v), 
            observed={'v': theta})

        # Initial parameters for the sampler
        t0_guess_idx = (np.abs(F - np.max(F))).argmin()

        # Initialization of the chain
        start = {'DeltaF':np.max(F), 'Fb':0.,
            't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}

        # Fit model with NUTS
        trace = pm.sample(2000, tune=2000, nuts_kwargs=dict(target_accept=.95),
        start=start)

    return trace 

for event_index, lightcurve in enumerate(lightcurves):

    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

    # Initialize Celerite model
    model_GP = PointSourcePointLensGP_emcee(t, F, sigF)

    # Create custom theano Op
    log_likelihood = LogLike(model_GP.log_likelihood, model_GP.gp, F)

    # Fit pymc3 model
    trace = fit_pymc3_model(t, F, sigF, log_likelihood)