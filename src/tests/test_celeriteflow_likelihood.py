import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import pymc3 as pm
import seaborn as sns
import os
import sys
sys.path.append("../codebase")
from data_preprocessing_ogle import process_data
from plotting_utils import plot_data, plot_emcee_traceplots
from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

import tensorflow as tf
session = tf.get_default_session()
if session is None:
    session = tf.InteractiveSession()

import theano
import theano.tensor as tt
#from theano.tests import unittest_tools as utt
import celeriteflow as cf
import celerite
from celerite.modeling import Model # an abstract class implementing the 
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

        mean_model = CustomCeleriteModel(0., 0., self.t[t0_guess_idx], 0.1, 10.)

        # Set up the GP model
        kernel = terms.Matern32Term(log_sigma=np.log(2.), log_rho=np.log(10))

        gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)

        self.gp = gp

    def log_likelihood(self, pars):    
        ln_sigma, ln_rho, DeltaF, Fb, t0, teff, tE = pars


        self.gp.compute(self.t, self.sigF)
        self.gp.set_parameter_vector((ln_sigma,ln_rho,DeltaF,Fb,t0,teff,tE))

        #print("Celerite parameters:", self.gp.get_parameter_dict())

        return self.gp.log_likelihood(self.F) 

def log_likelihood_celeriteflow(t, F, sigF, ln_sigma, ln_rho, DeltaF, Fb, t0, teff, 
        tE):

    # Evaluate forward model
    u0 = teff/tE
    u = tf.sqrt(u0**2 + ((t - t0)/tE)**2)
        
    A = lambda u: (u**2 + 2)/(u*tf.sqrt(u**2 + 4))  
    
    mean = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

    # Set up the gp
    kernel = cf.terms.Matern32Term(log_sigma=ln_sigma,
                            log_rho=ln_rho)

    # Evaluate log-likelihood
    solver = cf.Solver(kernel, t, sigF)
    alpha = solver.apply_inverse(F[:, None])

    # Print the covariance matrix
    log_det = solver.log_determinant

    # Add print operation
    
    log_likelihood = -0.5 * (
        tf.squeeze(
            tf.matmul(F[None, :] - mean, alpha))
        + solver.log_determinant
        + tf.cast(tf.size(t), tf.float64)*tf.constant(np.log(2*np.pi), 
            dtype=tf.float64))

    return log_likelihood, log_det


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

for event_index, lightcurve in enumerate(lightcurves):
    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

    # Calculate celerite likelihood
    celerite_model = PointSourcePointLensGP_emcee(t, F, sigF)

    default_params = np.array([1., 0.5, 10., 0., 7800., 15., 20.])

    celerite_model.gp.set_parameter_vector(default_params)
    celerite_likelihood = celerite_model.log_likelihood(default_params)

    # Calculate celeriteflow log-likelihood
    ln_sigma, ln_rho, DeltaF, Fb, t0, teff, tE = default_params

    print("Celeriteflow parameters: ", default_params)

    celeriteflow_likelihood, log_det = log_likelihood_celeriteflow(t, F, sigF, 
        ln_sigma, ln_rho, DeltaF, Fb, t0, teff, tE)


    print("Celerite likelihood: ", celerite_likelihood)

    # Add print operation
    printing = tf.Print(celeriteflow_likelihood, [celeriteflow_likelihood], 
        message="Celeriteflow likelihood: ")
    printing2 = tf.Print(log_det, [log_det], 
        message="Celeriteflow log determinant: ")

#    printing.eval()