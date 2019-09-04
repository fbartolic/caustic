import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from matplotlib import pyplot as plt

from .utils import construct_masked_tensor, estimate_t0

import exoplanet as xo
from exoplanet.gp import terms, GP

from scipy.special import zeta
from scipy.stats import invgamma
from scipy.optimize import fsolve

class SingleLensModel(pm.Model):
    """
    Abstract class for a single lens model.  Subclasses should implement the
    should implement the `magnification` method, and if needed, the
    `evaluate_posterior_model_on_grid`, `evaluate_prior_model_on_grid`, and
    `evaluate_map_model_on_grid` methods. All of the noise models are 
    implemented in this class. Subclasses implement forward models which 
    compute the magnification.

    Parameters
    ----------
    data : caustic.data 
        Caustic data object.
    errorbar_rescaling : str, optional
        Defines how the error bars are treated in models. Choose between 
        'none', 'constant', 'additive_variance', and  'flux_dependant', 
        by default 'constant'.
    kernel : str, optional
        Choose a GP kernel, options are 'white_noise' and 'matern32', by 
        default 'white_noise'.
    """
    #  override __init__ function from pymc3 Model class
    def __init__(
        self, 
        data=None, 
        errorbar_rescaling='constant', 
        kernel='white_noise'
    ):
        super(SingleLensModel, self).__init__()

        # Load data standardized to zero median and unit variance
        tables = data.get_standardized_data()
        self.n_bands = len(tables) # number of photometric bands

        # Useful attributes
        self.t_min = np.min([table['HJD'][0] for table in tables])
        self.t_max = np.max([table['HJD'][-1] for table in tables])
        self.max_npoints = int(np.max([len(table['HJD']) for table in tables]))

        # Construct tensors used for computing the models
        t_list = [np.array(table['HJD']) for table in tables]
        F_list = [np.array(table['flux']) for table in tables]
        sig_F_list = [np.array(table['flux_err']) for table in tables]

        self.t, self.mask = construct_masked_tensor(t_list)
        self.F, _ = construct_masked_tensor(F_list)
        self.sig_F, _ = construct_masked_tensor(sig_F_list)

        self.errorbar_rescaling = errorbar_rescaling
        self.kernel = kernel 

    def compute_magnification(self, u, u0):
        """
        Computes the magnification fraction [A(u) - 1]/[A(u0) - 1]
        where A(u) is the magnification function.
        
        Parameters
        ----------
        u : theano.tensor
            Trajectory of the lens u(t) with respect to the source in units of
            ang. Einstein radii.
        u0 : theano.tensor
            Lens-source separation at time t_.

        Returns
        -------
        theano.tensor
            The value of the magnification at each time t.
        """
        A_u = (u**2 + 2)/(u*T.sqrt(u**2 + 4))
        A_u0 = (u0**2 + 2)/(T.abs_(u0)*T.sqrt(u0**2 + 4))

        return (A_u - 1)/(A_u0 - 1)

    def __init_matern32_noise_model(self):
        """
        Initializes all parameters for the matern32 Gaussian Process
        noise model.

        """
        def solve_for_invgamma_params(params, x_min, x_max):
            """
            Returns parameters of an inverse zeta distribution p(x) such that 
            0.1% of total prob. mass is assigned to values of x < x_min and 
            1% of total prob. masss  to values greater than x_max.
            """
            lower_mass = 0.01
            upper_mass = 0.99

            # Trial parameters
            alpha, beta = params

            # Equation for the roots defining params which satisfy the constraint
            return (invgamma.cdf(x_min, alpha, scale=beta) - \
                lower_mass, invgamma.cdf(x_max, alpha, scale=beta) - upper_mass)

        # Compute parameters for the prior on GP hyperparameters
        invgamma_a = np.zeros(self.n_bands)
        invgamma_b = np.zeros(self.n_bands)

        for i in range(self.n_bands):
            t_ = self.t[i].eval()
            invgamma_a[i], invgamma_b[i] = fsolve(solve_for_invgamma_params, 
                (0.1, 0.1), (np.median(np.diff(t_)), t_[-1] - t_[0]))

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 

        self.sigma = BoundedNormal('sigma', 
            mu=T.zeros((self.n_bands, 1)),
            sd=3.*T.ones((self.n_bands, 1)),
            testval=0.5*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        self.rho = pm.InverseGamma('rho', 
            alpha = T.as_tensor_variable(invgamma_a)*T.ones((self.n_bands, 1)),
            beta = T.as_tensor_variable(invgamma_b)*T.ones((self.n_bands, 1)),
            testval=2.*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))
        
    def compute_log_likelihood(self, r, var_F, kernel=None):
        """"
        Computes the total log likelihood of the model assuming that the 
        observations in different bands are independent. 
        The likelihood is assumed to be multivariate Gaussian with an optional
        Gaussian Process modeling the covariance matrix elements.

        Parameters
        ----------
        r : theano.tensor
            Residuals of the mean model with respect to the data.
        var_F : theano.tensor
            Diagonal elements of the covariance matrix. 

        Returns
        -------
        theano.tensor 
            Log likelihood.
        """
        ll = 0 
        # Iterate over all observed bands
        for i in range(self.n_bands):
            t = self.t[i][self.mask[i].nonzero()]
            r_ = r[i][self.mask[i].nonzero()]

            if (kernel==None):
                ll += -0.5*T.sum(r_**2/var_F) -\
                    0.5*T.sum(T.log(2*np.pi*var_F))

            elif (kernel=='matern32'):
                kernel = terms.Matern32Term(sigma=self.sigma, rho=self.rho)
                gp = GP(kernel, t, var_F, J=2) # J=2 for matern32 kernel
                ll += gp.log_likelihood(r_)
            
            else:
                raise ValueError('Specified kernel not recognized.')
        
        return ll
    
    def compute_marginalized_log_likelihood(self, magnification, var_F):
        """
        Computes log likelihood marginalized over the linear flux parameters.
        """
#        def log_likelihood_single_band(F, var_F, mag, mask):
#        F = F[mask.nonzero()]
#        var_F = var_F[mask.nonzero()]
#        mag = mag[mask.nonzero()]

        T.printing.Print('magnification shape')(T.shape(magnification))


        F = self.F[0][self.mask[0].nonzero()]
        var_F = var_F[0][self.mask[0].nonzero()]
        mag = magnification[0][self.mask[0].nonzero()]

        T.printing.Print('F shape')(T.shape(F))
        T.printing.Print('var_F shape')(T.shape(var_F))
        T.printing.Print('mag shape')(T.shape(mag))
        T.printing.Print('mask shape')(T.shape(self.mask))
        T.printing.Print('mag values')(mag)

        N = T.shape(F)[0]
        # Linear parameter matrix
#        mu_theta = T.dot(mag_vector, np.max(self.F))
        A = T.stack([mag, T.ones_like(F)], axis=1)

        # Covariance matrix
        C_diag = var_F
        C = T.nlinalg.diag(C_diag)

        # Prior matrix
        sigDelta_F = 10.
        sig_F_base = 0.1
        L_diag = T.as_tensor_variable(np.array([sigDelta_F, sig_F_base])**2.)
        L = T.nlinalg.diag(L_diag)

        T.printing.Print('got this far')(T.shape(F))

        # Calculate inverse of covariance matrix for marginalized likelihood
        inv_C = T.nlinalg.diag(T.pow(C_diag, -1.))
        ln_detC = T.log(C_diag).sum()

        inv_L = T.nlinalg.diag(T.pow(L_diag, -1.))
        ln_detL = T.log(L_diag).sum()

        S = inv_L + T.dot(A.transpose(), T.dot(inv_C, A))
        inv_S = T.nlinalg.matrix_inverse(S)
        ln_detS = T.log(T.nlinalg.det(S))

        inv_SIGMA =inv_C -\
            T.dot(inv_C, T.dot(A, T.dot(inv_S, T.dot(A.transpose(), inv_C))))
        ln_detSIGMA = ln_detC + ln_detL + ln_detS

#        # Calculate marginalized likelihood
        r = F #- mu_theta
        ll = -0.5*T.dot(r.transpose(), T.dot(inv_SIGMA, r)) -\
            0.5*N*np.log(2*np.pi) - 0.5*ln_detSIGMA

        T.printing.Print('success')(ll)
        
        return ll
#
#        # Compute the log-likelihood which is additive across different bands
#        ll = 0 
#        for i in range(self.n_bands):
#            ll += log_likelihood_single_band(self.F[i], 
#                var_F[i], magnification[i], self.mask[i])
#
#        return ll 

    def generate_mock_dataset(self):
        """
        Generates mock caustic.data object by sampling the prior predictive 
        distribution.
        """
        raise NotImplementedError()

    def sample_with_emcee(self, n_walkers=50, n_samples=10000, start=None):
        """
        Samples the model posterior distribution using emcee.
                
        Parameters
        ----------
        output_dir : str
            Path to directory where the trace is to be saved, together with
            various diagnostic plots and information. If it is not defined 
            nothing will be saved.
        n_walkers: int, optional
            Number of walkers, by default 50.
        n_samples : int, optional
            Number of sampling steps, by default 10000.
        start: dict, optional
            Initial point in the parameter space at which the tuning steps
            start.

        Returns
        -------
        ndarray
            Chain containing posterior samples of shape 
        """
        # Print the names of the free parameters and the inital values
        # of their log-priors        
        free_parameters = [RV.name for RV in self.basic_RVs]
        initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]
        if np.any(np.isnan(initial_logps))==True:
            print("Prior distributions misspecified, check that the test\
                values are within the bounds of the prior.")

        print("Free parameters:\n", free_parameters)

        # DFM's hack for using emcee with PyMC3 models
        f = theano.function(self.vars, [self.logpt] + self.deterministics)
    
        def log_prob_func(params):
            dct = self.bijection.rmap(params[::-1])
            args = (dct[k.name] for k in self.vars)
            results = f(*args)
            return tuple(results)

        # First we work out the shapes of all of the deterministic variables
        initial_params = pm.find_MAP()

        # If custom initial parameters are specified, update relevant parameters
        if start is not None:
            for key in initial_params.keys():
                if key in start.keys():
                    initial_params[key] = np.array([start[key]])

        # For some reason, bijection.map flips the ordering of the variables
        # from that in self.vars, hence [::-1]
        vec = self.bijection.map(initial_params)[::-1]
        initial_blobs = log_prob_func(vec)[1:]
        dtype = [(var.name, float, np.shape(b)) for var,
             b in zip(self.deterministics, initial_blobs)]
        
        # Then sample as usual
        coords = vec + 1e-5 * np.random.randn(n_walkers, len(vec))
        nwalkers, ndim = coords.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func, 
            blobs_dtype=dtype)
        sampler.run_mcmc(coords, n_samples, progress=True)