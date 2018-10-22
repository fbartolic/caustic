import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
import corner
import pymc3 as pm
import seaborn as sns
import os
from codebase.data_preprocessing_ogle import process_data
from codebase.plotting_utils import plot_data, plot_emcee_traceplots
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

def solve_for_invgamma_params(params, t_min, t_max):
    def inverse_gamma_cdf(x, alpha, beta):
        return invgamma.cdf(x, alpha, scale=beta)

    alpha, beta = params

    return (inverse_gamma_cdf(2*t_min, alpha, beta) - \
    0.001, inverse_gamma_cdf(t_max, alpha, beta) - 0.99)

def _to_tensor_type(shape):
    return tt.TensorType(dtype="float64", broadcastable=[False]*len(shape))

class TensorFlowOp(tt.Op):
    """A custom Theano Op uses TensorFlow as the computation engine
    
    Args:
        target (Tensor): The TensorFlow tensor defining the output of
            this operation
        parameters (list(Tensor)): A list of TensorFlow tensors that
            are inputs to this operation
        names (Optional(list)): A list of names for the parameters.
            These are the names that will be used within PyMC3
        feed_dict (Optional(dict)): A "feed_dict" that is provided to
            the TensorFlow session when the operation is executed
        session (Optional): A TensorFlow session that can be used to
            evaluate the operation
    
    """
    def __init__(self, target, parameters, names=None, feed_dict=None, session=None):
        self.parameters = parameters
        self.names = names
        self._feed_dict = dict() if feed_dict is None else feed_dict
        self._session = session
        self.target = target
        
        # Execute the operation once to work out the shapes of the
        # parameters and the target
        in_values, out_value = self.session.run(
            [self.parameters, self.target], feed_dict=self._feed_dict)
        self.shapes = [np.shape(v) for v in in_values]
        self.output_shape = np.shape(out_value)
        
        # Based on this result, work out the shapes that the Theano op
        # will take in and return
        self.itypes = tuple([_to_tensor_type(shape) for shape in self.shapes])
        self.otypes = tuple([_to_tensor_type(self.output_shape)])
        
        # Build another custom op to represent the gradient (see below)
        self._grad_op = _TensorFlowGradOp(self)

    @property
    def session(self):
        """The TensorFlow session associated with this operation"""
        if self._session is None:
            self._session = tf.get_default_session()
        return self._session
    
    def get_feed_dict(self, sample):
        """Get the TensorFlow feed_dict for a given sample
        
        This method will only work when a value for ``names`` was provided
        during instantiation.
        
        sample (dict): The specification of a specific sample in the chain
        
        """
        if self.names is None:
            raise RuntimeError("'names' must be set in order to get the feed_dict")
        return dict(((param, sample[name])
                     for name, param in zip(self.names, self.parameters)),
                    **self._feed_dict)
    
    def infer_shape(self, node, shapes):
        """A required method that returns the shape of the output"""
        return self.output_shape,

    def perform(self, node, inputs, outputs):
        """A required method that actually executes the operation"""
        # To execute the operation using TensorFlow we must map the inputs from
        # Theano to the TensorFlow parameter using a "feed_dict"
        feed_dict = dict(zip(self.parameters, inputs), **self._feed_dict)
        outputs[0][0] = np.array(self.session.run(self.target, feed_dict=feed_dict))

    def grad(self, inputs, gradients):
        """A method that returns Theano op to compute the gradient
        
        In this case, we use another custom op (see the definition below).
        
        """
        op = self._grad_op(*(inputs + gradients))
        # This hack seems to be required for ops with a single input
        if not isinstance(op, (list, tuple)):
            return [op]
        return op

class _TensorFlowGradOp(tt.Op):
    """A custom Theano Op defining the gradient of a TensorFlowOp
    
    Args:
        base_op (TensorFlowOp): The original Op
    
    """
    def __init__(self, base_op):
        self.base_op = base_op
        
        # Build the TensorFlow operation to apply the reverse mode
        # autodiff for this operation
        # The placeholder is used to include the gradient of the
        # output as a seed
        self.dy = tf.placeholder(tf.float64, base_op.output_shape)
        self.grad_target = tf.gradients(base_op.target,
                                        base_op.parameters,
                                        grad_ys=self.dy)

        # This operation will take the original inputs and the gradient
        # seed as input
        types = [_to_tensor_type(shape) for shape in base_op.shapes]
        self.itypes = tuple(types + [_to_tensor_type(base_op.output_shape)])
        self.otypes = tuple(types)
 
    def infer_shape(self, node, shapes):
        return self.base_op.shapes

    def perform(self, node, inputs, outputs):
        feed_dict = dict(zip(self.base_op.parameters, inputs[:-1]),
                         **self.base_op._feed_dict)
        feed_dict[self.dy] = inputs[-1]
        result = self.base_op.session.run(self.grad_target, feed_dict=feed_dict)
        for i, r in enumerate(result):
            outputs[i][0] = np.array(r)


def log_likelihood(t, F, sigF, DeltaF, Fb, t0, teff, tE, u_K):
    """Custom log-likelihood implemented in tensorflow.
    
    Parameters
    ----------
    t : tf tensor
        Time array. 
    
    F : tf tensor
        Flux tensor. 

    sigF : tensor
        Flux errors. 

    params : list(tf tensor)

    Returns 
    -------
    log_likelihood : tf tensor
        The log-likelihood at given parameters. 
    """

   # DeltaF, Fb, t0, teff, tE = params

    # Evaluate forward model
    u0 = teff/tE
    u = tf.sqrt(u0**2 + ((t - t0)/tE)**2)
        
    A = lambda u: (u**2 + 2)/(u*tf.sqrt(u**2 + 4))  
    
    mean = DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb

    K = tf.cond(tf.less(u_K, 0.), lambda: tf.cast(1., dtype=tf.float64),  
        lambda: 1. - tf.log(1 - u_K))

    # Evaluate log-likelihood
    return -0.5 * tf.reduce_sum(tf.square((F - mean)/(K*sigF)))

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

def fit_pymc3_model(t, F, sigF):
    # First, expose the TensorFlow log likelihood implementation to Theano
    # so that PyMC3 can use it
    # NOTE: The "names" parameter refers to the names that will be used in
    # in the PyMC3 model (see below)
    DeltaF_tens = tf.Variable(1.0, dtype=tf.float64, name="DeltaF")
    Fb_tens = tf.Variable(0.0, dtype=tf.float64, name="Fb")
    t0_tens = tf.Variable(100.0, dtype=tf.float64, name="t0")
    teff_tens = tf.Variable(10.0, dtype=tf.float64, name="teff")
    tE_tens = tf.Variable(20.0, dtype=tf.float64, name="tE")
    u_K_tens = tf.Variable(0.40, dtype=tf.float64, name="u_K")
    #ln_sigma_tens = tf.Variable(0.0, dtype=tf.float64, name="ln_sigma")
    #ln_rho_tens = tf.Variable(0.0, dtype=tf.float64, name="ln_rho")
    
    session.run(tf.global_variables_initializer())

    log_likelihood_tensor = log_likelihood(t, F, sigF, DeltaF_tens, Fb_tens,
        t0_tens, teff_tens, tE_tens, u_K_tens)

    tf_loglike = TensorFlowOp(log_likelihood_tensor, [DeltaF_tens, Fb_tens, t0_tens, 
        teff_tens, tE_tens, u_K_tens],
                          names=["DeltaF", "Fb", "t0", "teff", "tE", "u_K"])
    #                       "ln_sigma", "log_rho"])

    # Test the gradient
    pt = session.run(tf_loglike.parameters)
    #utt.verify_grad(tf_loglike, pt)
    
    model = pm.Model()

    # Compute parameters for the prior on GP hyperparameters
    invgamma_a, invgamma_b =  fsolve(solve_for_invgamma_params, (0.1, 0.1), 
        (np.median(np.diff(t)), t[-1] - t[0]))

    with model:    
        # log density of the joint prior for (tE, teff)
        def joint_density(value):
            teff = tt.cast(value[0], 'float64')
            tE = tt.cast(value[1], 'float64')
            sig_tE = tt.cast(365., 'float64')
            sig_u0 = tt.cast(1., 'float64')
            return -tt.log(tE) - (teff/tE)**2/sig_u0**2 - tE**2/sig_tE**2

        # Priors for GP hyperparameters
        def ln_rho_prior(ln_rho):
            lnpdf_lninvgamma = lambda  x, a, b: np.log(x) + a*np.log(b) -\
                 (a + 1)*np.log(x) - b/x - np.log(gamma(a)) 

            res = lnpdf_lninvgamma(np.exp(ln_rho), invgamma_a, invgamma_b)
            return tt.cast(res, 'float64')

        #ln_rho = pm.DensityDist('ln_rho', ln_rho_prior, testval = 0.)

        def ln_sigma_prior(ln_sigma):
            sigma = np.exp(ln_sigma)
            res = np.log(sigma) - sigma**2/3.**2
            return tt.cast(res, 'float64')

        #ln_sigma = pm.DensityDist('ln_sigma', ln_sigma_prior, testval = 2.)

        # Priors for unknown model parameters
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) # DeltaF is positive
        DeltaF = BoundedNormal('DeltaF', mu=np.max(F), sd=1.)
        Fb = pm.Normal('Fb', mu=0., sd=0.1)
        t0 = pm.Uniform('t0', 0, 1.)
        teff_tE = pm.DensityDist('teff_tE', joint_density, shape=2, 
            testval = [0.1, 10.])
        u_K = pm.Uniform('u_K', -1., 1.)

        # Transform t0 to sensible units
        t0 = (t[-1] - t[0])*t0 + t[0]

        teff = teff_tE[0]
        tE = teff_tE[1]

        # Define a custom "potential" to calculate the log likelihood
        pm.Potential("loglike", tf_loglike(DeltaF, Fb, t0, teff, tE, u_K))
       
        # Initial parameters for the sampler
        t0_guess_idx = (np.abs(F - np.max(F))).argmin()

        # Initialization of the chain
        start = {'DeltaF':np.max(F), 'Fb':0.,
            't0':(t[t0_guess_idx] - t[0])/(t[-1] - t[0])}

        # Fit model with NUTS
        trace = pm.sample(2000, tune=2000, cores=1, 
            nuts_kwargs=dict(target_accept=.95), start=start)

    return trace 

for event_index, lightcurve in enumerate(lightcurves):

    # Pre process the data
    t, F, sigF = process_data(lightcurve[:, 0], lightcurve[:, 1], 
        lightcurve[:, 2], standardize=True)

    # Fit pymc3 model
    trace = fit_pymc3_model(t, F, sigF)