import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from matplotlib import pyplot as plt

import exoplanet as xo
from exoplanet.gp import terms, GP

from scipy.special import zeta
from scipy.stats import invgamma
from scipy.optimize import fsolve

from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy import units as u

import emcee
import corner

class SingleLensModel(pm.Model):
    """
    Abstract class for a single lens model.  Subclasses should implement the
    should implement the `magnification` method, and if needed, the
    `evaluate_posterior_model_on_grid`, `evaluate_prior_model_on_grid`, and
    `evaluate_map_model_on_grid` methods. All of the noise models are 
    implemented in this class. Subclasses implement forward models which 
    compute the magnification.
    """
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, errorbar_rescaling='constant', kernel='white_noise'):
        """
        Parameters
        ----------
        data : caustic.data 
            Caustic data object.
        errorbar_rescaling : str, optional
            Defines how the error bars are treated in models. Choose between 
            'none', constant', 'additive_variance', and  'flux_dependant', 
            by default 'constant'.
        kernel : str, optional
            Choose a GP kernel, options are 'white_noise' and 'matern32', by 
            default 'white_noise'.
        """
        super(SingleLensModel, self).__init__()

        # Load and rescale the data to zero median and unit variance
        tables = data.get_standardized_data()
        self.n_bands = len(tables) # number of photometric bands

        # Useful attributes
        self.t_begin = np.min([table['HJD'][0] for table in tables])
        self.t_end = np.max([table['HJD'][-1] for table in tables])
        self.max_npoints = int(np.max([len(table['HJD']) for table in tables]))
        self.fluxes_parametrization = 'dominik2009'

        # Construct tensors used for computing the models
        t_list = [np.array(table['HJD']) for table in tables]
        F_list = [np.array(table['flux']) for table in tables]
        sigF_list = [np.array(table['flux_err']) for table in tables]

        self.t, self.mask = self.construct_masked_tensor(t_list)
        self.F, _ = self.construct_masked_tensor(F_list)
        self.sigF, _ = self.construct_masked_tensor(sigF_list)

        self.errorbar_rescaling = errorbar_rescaling
        self.kernel = kernel 

        # Define bounded normal prior distributions which are often needed
        self.BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        self.BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize noise model
        self._compute_varF()
        if (kernel=='white_noise'):
            pass
        elif (kernel=='matern32'):
            self._init_matern32_noise_model()
        else:
            raise ValueError('Kernel not recognized.')

    
    def construct_masked_tensor(self, array_list):
        """
        Given a list of 1D numpy arrays, this function returns a theano tensor
        of shape (n_elements, n_max) where n_elements is the number of arrays
        in the list, and n_max is the length of the largest array in the list.
        The missing values are filled in with zeros and a mask is returned 
        together with tensor. The purpose of this function is to construct 
        tensors instead of using lists in order to avoid having to do loops.

        Parameters
        ----------
        array_list : list 
            List of numpy arrays of varying lengths. 
        
        Returns
        -------
        tuple 
            Returns tuple (tensor, mask) where tensor and maks are 
            theano.tensor objects of the same shape (n_elements, n_max). 
            The mask tensor is of datta type int8 and the elements are 
            equal to 1 for non-filled in values and zero otherwise. To use
            the mask in theano, use `tensor[mask.nonzero()]`.

        """
        for array in array_list:
            if(array.ndim != 1):
                raise ValueError('Make sure that all of the arrays are 1D and\
                have the same dimension')

        n_max =  np.max([len(array) for array in array_list])
        tensor = T._shared(np.stack([np.pad(array, 
            (0, n_max - len(array)), 'constant', 
            constant_values=(0.,)) for array in array_list]))
        
        masks_list = []

        for array in array_list:
            array = np.append(
                np.ones(len(array)),  
                np.zeros(n_max - len(array))
                )
            masks_list.append(array)

        mask = T._shared(np.stack(masks_list).astype('int8'))

        return tensor, mask

    def magnification(self, t, u0):
        """
        Calculates the magnification fraction [A(u) - 1]/[A(u0) - 1]
        where A(u) is the magnification function.
        """
        pass
    
    def initialize_linear_parameters(self):
        """
        Initializes linear flux parameters Delta_F and F_base. The total 
        number of linear parameters is then 2*n_bands.
        """
        if (self.fluxes_parametrization=='dominik2009'):
            # Initialize linear parameters
            self.Delta_F = self.BoundedNormal('Delta_F', 
                mu=T.zeros((self.n_bands, 1)),
                sd=50.*T.ones((self.n_bands, 1)),
                testval=5.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.F_base = pm.Normal('F_base', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                testval=T.zeros((self.n_bands, 1)),
                shape=(self.n_bands, 1))
        elif (self.fluxes_parametrization=='standard'):
            self.FS = pm.Normal('F_S', 
                mu=T.zeros((self.n_bands, 1)),
                sd=10.*T.ones((self.n_bands, 1)),
                testval=3.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.FB = pm.Normal('F_B', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=10*T.ones((self.n_bands, 1)),
                testval=T.zeros((self.n_bands, 1)),
                shape=(self.n_bands, 1))

    def t0_guess(self, event):
        """
        Guesses an intial value for the t0 parameter. This is necessary because
        the posterior is highly multi-modal in t0 and the sampler takes
        ages to converge if t0 is not close to true value.
        """
        tmp = event.masks.copy()
        event.remove_worst_outliers(window_size=20, mad_cutoff=2)
        tables = event.get_standardized_data()
        fluxes = np.concatenate([table['flux'] for table in tables])
        times = np.concatenate([table['HJD'] for table in tables])
        event.masks = tmp

        guess = np.median(times[fluxes > 4])

        cut = 4
        while np.isnan(guess):
            guess = np.median(times[fluxes > cut])
            cut -= 0.5

        return guess

    def _compute_varF(self):
        """
        Computes the modeled variance (squared "error-bars") of the observed 
        fluxes. The user can can choose between three different options. The
        first is a rescaling all error bars by a constant factor A. The second
        is to is same as the first but with an additional additive variance 
        term. The third option is the same as the second but with the additive
        variance term is proportional to the magnification, reaching a maximum
        value at highest magnification and it vanishes otherwise.

        """
        if (self.errorbar_rescaling=='none'):
            # Diagonal terms of the covariance matrix
            self.varF = self.sigF**2

        elif (self.errorbar_rescaling=='constant'):
            ## Noise model parameters
            self.A = self.BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            # Diagonal terms of the covariance matrix
            self.varF = (self.A*self.sigF)**2

        elif (self.errorbar_rescaling=='additive_variance'):
            ## Noise model parameters
            self.A = self.BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = self.BoundedNormal('B', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=1*T.ones((self.n_bands, 1)),
                testval=0.01*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))
            
            # Diagonal terms of the covariance matrix
            self.varF = (self.A*self.sigF)**2 + self.B**2

        elif (self.errorbar_rescaling=='flux_dependant'):
            ## Noise model parameters
            self.A = self.BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = self.BoundedNormal('B', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=5*T.ones((self.n_bands, 1)),
                testval=0.01*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            # Diagonal terms of the covariance matrix
            self.varF = (self.A*self.sigF)**2 +\
                 (self.magnification(self.t)*self.B)**2

        else:
            raise ValueError('Specified option for rescaling the flux error\
                bars not recognized.')

    def _init_matern32_noise_model(self):
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

        self.BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        self.sigma = self.BoundedNormal('sigma', 
            mu=T.zeros((self.n_bands, 1)),
            sd=3.*T.ones((self.n_bands, 1)),
            testval=0.5*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        self.rho = pm.InverseGamma('rho', 
            alpha = T._shared(invgamma_a)*T.ones((self.n_bands, 1)),
            beta = T._shared(invgamma_b)*T.ones((self.n_bands, 1)),
            testval=2.*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))
        
    def log_likelihood(self):
        """"
        Computes the total log likelihood of the model assuming that the 
        observations in different bands are independent. The choice of 
        the likelihood function is specified when initializing the model.
        """
        # Compute the model residuals
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base 

        self.r = self.F - mean_func

        ll = 0 
        # Iterate over all observed bands
        for i in range(self.n_bands):
            t = self.t[i][self.mask[i].nonzero()]
            r = self.r[i][self.mask[i].nonzero()]
            varF = self.varF[i][self.mask[i].nonzero()]

            if (self.kernel=='white_noise'):
                ll += -0.5*T.sum(r**2/varF) -\
                    0.5*T.sum(T.log(2*np.pi*varF))

            elif (self.kernel=='matern32'):
                kernel = terms.Matern32Term(sigma=self.sigma, rho=self.rho)
                gp = GP(kernel, t, varF, J=2) # J=2 for matern32 kernel
                ll += gp.log_likelihood(r)
            
            else:
                raise ValueError('Kernel choice not recognized.')

        pm.Potential('log_likelihood', ll)
    
    def evaluate_posterior_model_on_grid(self, trace, t_grids, n_samples=50):
        """
        Evaluates a posterior model on a dense grid given samples from a 
        posterior distribution over the model parameters.
        
        Parameters
        ----------
        trace : PyMC3 MultiTrace object
            Trace object containing samples from posterior.
        t_grids : list of 1D numpy arrays
            Time grids on which the model is to be evaluated. The list should
            contain one array for each observing band.
        n_samples : int, optional
            Number of parameter samples for which the model is to be evaluated,
            these are randomly picked from the trace object, by default 50.

        Returns
        -------
        list
            List of numpy arrays of shape (n_sample, len(t_grid)) for
            each t_grid in t_grids. Each element of the list corresponds to 
            a model prediction in a different band.
        """
        n_max =  np.max([len(t_grid) for t_grid in t_grids])
        prediction_eval = np.zeros((n_samples, self.n_bands, n_max))

        # Construct tensors needed to evaluate the model
        t_grids_tensor, mask = self.construct_masked_tensor(t_grids)

        # Evaluate model for each sample
        for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                size=n_samples)):
            
            if (self.kernel=='white_noise'):
                # Tensor which is to be evaluated in model context
                prediction = self.Delta_F*self.magnification(t_grids_tensor) +\
                    self.F_base

                prediction_eval[i] = xo.eval_in_model(prediction, sample)

            elif (self.kernel=='matern32'):
                # Tensor which is to be evaluated in model context
                prediction_mean = self.Delta_F*self.magnification(t_grids_tensor) +\
                    self.F_base
                prediction_mean_obs = self.Delta_F*self.magnification(self.t) +\
                    self.F_base

                for n in range(self.n_bands):
                    # Initialize kernel and gp object
                    kernel = terms.Matern32Term(sigma=self.sigma[n], 
                        rho=self.rho[n])

                    gp = GP(kernel, self.t[n], self.varF[n], J=2)

                    # Calculate log_likelihood 
                    r = self.F[n] - prediction_mean_obs[n]
                    gp.log_likelihood(r)

                    prediction_eval[i, n] =\
                        xo.eval_in_model(gp.predict(t_grids[n]), sample) 

                # Add mean model to GP prediction
                prediction_eval[i] += xo.eval_in_model(prediction_mean, sample)

        # Construct list of prediction_eval
        mask_numpy  = mask.eval().astype(bool)

        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[:, i, mask_numpy[i, :]])

        return pred_list

    def evaluate_prior_model_on_grid(self, trace, t_grids):
        """
        Evaluates prior model given samples from prior distribution.
        
        Parameters
        ----------
        trace : dict
            Dictionary with variable names as keys. The values are numpy 
            arrays (of varying shapes depending on the parameter) containing 
            prior samples. 
        t_grids : list of 1D numpy arrays
            Time grids on which the model is to be evaluated. The list should
            contain one array for each observing band.
        
        Returns
        -------
        list
            List of numpy arrays of shape (n_sample, len(t_grid)) for
            each t_grid in t_grids. Each element of the list corresponds to 
            a model prediction in a different band.
        """
        n_samples = len(np.atleast_1d(trace['t0']))
        n_max =  np.max([len(t_grid) for t_grid in t_grids])
        prediction_eval = np.zeros((n_samples, self.n_bands, n_max))

        # Construct tensors needed to evaluate the model
        t_grids_tensor, mask = self.construct_masked_tensor(t_grids)

        # Evaluate  model for each sample
        for i in range(n_samples):
            if (n_samples==1):
                sample = {key:trace[key] for key in trace.keys()}
            else:
                sample = {key:trace[key][i] for key in trace.keys()}
            
            if (self.kernel=='white_noise'):
                # Tensor which is to be evaluated in model context
                prediction = self.Delta_F*self.magnification(t_grids_tensor) +\
                    self.F_base

                prediction_eval[i] = xo.eval_in_model(prediction, sample)

            elif (self.kernel=='matern32'):
                # Tensor which is to be evaluated in model context
                prediction_mean = self.Delta_F*self.magnification(t_grids_tensor) +\
                    self.F_base
                prediction_mean_obs = self.Delta_F*self.magnification(self.t) +\
                    self.F_base

                for n in range(self.n_bands):
                    # Initialize kernel and gp object
                    kernel = terms.Matern32Term(sigma=self.sigma[n], 
                        rho=self.rho[n])

                    gp = GP(kernel, self.t[n], self.varF[n], J=2)

                    # Calculate log_likelihood 
                    r = self.F[n] - prediction_mean_obs[n]
                    gp.log_likelihood(r)

                    prediction_eval[i, n] =\
                        xo.eval_in_model(gp.predict(t_grids[n]), sample) 

                # Add mean model to GP prediction
                prediction_eval[i] += xo.eval_in_model(prediction_mean, sample)

        # Construct list of prediction_eval
        mask_numpy  = mask.eval().astype(bool)

        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[:, i, mask_numpy[i, :]])

        return pred_list

    def evaluate_map_model_on_grid(self, t_grids, map_point):
        """
        Evaluates a MAP model on a dense grid given the MAP parameters of
        the model.

        Parameters
        ----------
        t_grids : list of 1D numpy arrays
            Time grids on which the model is to be evaluated. The list should
            contain one array for each observing band.
        map_point : dict
            A dictionary contatining values of MAP parameters.

        Returns
        -------
        list
            List of numpy arrays of shape (len(t_grid)) for
            each t_grid in t_grids. Each element of the list corresponds to 
            a model MAP prediction in a different band.
        """
        n_max =  np.max([len(t_grid) for t_grid in t_grids])
        prediction_eval = np.zeros((self.n_bands, n_max))

        # Construct tensors needed to evaluate the model
        t_grids_tensor, mask = self.construct_masked_tensor(t_grids)

        if (self.kernel=='white_noise'):
            # Tensor which is to be evaluated in model context
            prediction = self.Delta_F*self.magnification(t_grids_tensor) +\
                self.F_base
            prediction_eval = xo.eval_in_model(prediction, map_point)

        elif (self.kernel=='matern32'):
            # Tensor which is to be evaluated in model context
            prediction_mean = self.Delta_F*self.magnification(t_grids_tensor) +\
                self.F_base
            prediction_mean_obs = self.Delta_F*self.magnification(self.t) +\
                self.F_base

            for n in range(self.n_bands):
                # Initialize kernel and gp object
                kernel = terms.Matern32Term(sigma=self.sigma[n], 
                    rho=self.rho[n])

                gp = GP(kernel, self.t[n], self.varF[n], J=2)

                # Calculate log_likelihood 
                r = self.F[n] - prediction_mean_obs[n]
                gp.log_likelihood(r)

                prediction_eval[n] =\
                    xo.eval_in_model(gp.predict(t_grids[n]), map_point) 

            # Add mean model to GP prediction
            prediction_eval += xo.eval_in_model(prediction_mean, map_point)

        # Construct list of prediction_eval
        mask_numpy  = mask.eval().astype(bool)

        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[i, mask_numpy[i, :]])

        return pred_list
    
    def evaluate_trajectory_on_grid(self, trace, t_grid, n_samples=50):
        """
        Evaluates different trajectories of the lens w.r. top the source on 
        the plane of the sky in (u_n, u_e) coordinates.
        
        Parameters
        ----------
        trace : PyMC3 MultiTrace object
            Trace object containing samples from posterior.
        t_grid : 1D numpy array
            Time grid for which the model is to be evaluated.         
        n_samples : int, optional
            Number of parameter samples for which the model is to be evaluated,
            these are randomly picked from the trace object, by default 50.

        Returns
        -------
        ndarray
            Numpy array of shape (n_sample, 2, len(t_grid)). The first 
            component is u_n, the second u_e.
        """
        prediction_eval = np.zeros((n_samples, 2, len(t_grid)))

        # Construct tensors needed to evaluate the model
        t_grid_tensor, _ = self.construct_masked_tensor([t_grid])

        # Evaluate model for each sample
        for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                size=n_samples)):
            
            # Tensor which is to be evaluated in model context
            t = t_grid_tensor
            delta_zeta_n, delta_zeta_e = self.compute_delta_zeta(t_grid_tensor)
            if (self.parametrization=='angle_magnitude'):
                u_n = -self.u0*T.sin(self.psi) + (t - self.t0)/\
                    self.tE*T.cos(self.psi) + self.pi_E*delta_zeta_n
                u_e = self.u0*T.cos(self.psi) + (t - self.t0)/\
                    self.tE*T.sin(self.psi) + self.pi_E*delta_zeta_e
        
            elif (self.parametrization=='local_acceleration'):
                # Compute u(t)
                cospsi = (self.a_par*self.delta_zeta_n_ddot_t0 +\
                    self.a_per*self.delta_zeta_e_ddot_t0)/(self.delta_zeta_e_ddot_t0**2 +\
                        self.delta_zeta_n_ddot_t0**2)/self.pi_E
                sinpsi = (self.a_par*self.delta_zeta_e_ddot_t0 -\
                    self.a_per*self.delta_zeta_n_ddot_t0)/(self.delta_zeta_e_ddot_t0**2 +\
                        self.delta_zeta_n_ddot_t0**2)/self.pi_E

                u_n = -self.u0*sinpsi + (t - self.t0)/\
                    self.tE*cospsi + self.pi_E*delta_zeta_n
                u_e = self.u0*cospsi + (t - self.t0)/\
                    self.tE*sinpsi + self.pi_E*delta_zeta_e

            else: 
                cospsi = self.pi_EN/self.pi_E
                sinpsi = self.pi_EE/self.pi_E
                u_n = -self.u0*sinpsi + (t - self.t0)/\
                    self.tE*cospsi + self.pi_E*delta_zeta_n
                u_e = self.u0*cospsi + (t - self.t0)/\
                    self.tE*sinpsi + self.pi_E*delta_zeta_e

            prediction_eval[i, 0, :] = xo.eval_in_model(u_e, sample)
            prediction_eval[i, 1, :] = xo.eval_in_model(u_n, sample)

        return prediction_eval

    def generate_mock_dataset(self):
        """
        Generates mock caustic.data object by sampling the prior predictive 
        distribution.
        """
        raise NotImplementedError()

    def sample(self, output_dir=None, n_tune=2000, n_samples=2000, 
            target_accept=0.9, start=None):
        """
        Samples the model posterior distribution using a modified NUTS sampler
        from the 'exoplanet' package with support for dense mass matrices. 
        Sampling consists of several tuning steps, in each step an empirical 
        estimate of the parameter covariance matrix is constructed from the 
        chains and a linear transformation is applied to the parameter space to
        remove some of the correlations. Since Hamiltonian Monte Carlo isn't 
        invariant to affine transformation (as opposed to emcee which is), 
        this helps remove some of the correlations in the posterior and 
        usually speeds up sampling by at least an order of magnitude. For more
        details see Dan's blog post: https://dfm.io/posts/pymc3-mass-matrix/.
        
        Parameters
        ----------
        output_dir : str
            Path to directory where the trace is to be saved, together with
            various diagnostic plots and information. If it is not defined 
            nothing will be saved.
        n_tune : int, optional
            Number of tuning steps, by default 2000
        n_samples : int, optional
            Number of sampling steps, by default 2000
        target_accept : float, optional
            Target acceptance fraction for the NUTS sampler, high
            The step size for the leapfrog integrator in NUTS is tuned such 
            to approximate this acceptance rate. Higher values like 0.9 or 
            0.95 often work better for problematic posteriors, by default 0.9.
        start: dict, optional
            Initial point in the parameter space at which the tuning steps
            start.

        Returns
        -------
        PyMC3 MultiTrace object
            Trace object containing the posterior samples.
        """

        sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

        # Print the names of the free parameters and the inital values
        # of their log-priors        
        free_parameters = [RV.name for RV in self.basic_RVs]
        initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]
        if np.any(np.isnan(initial_logps))==True:
            print("Prior distributions misspecified, check that the test\
                values are within the bounds of the prior.")

        print("Free parameters:\n", free_parameters)

        # Run burn-in chains             
        burnin = sampler.tune(tune=n_tune,
                step_kwargs=dict(target_accept=target_accept), start=start)

        # Run final sampling
        trace = sampler.sample(draws=n_samples,
                step_kwargs=dict(target_accept=target_accept))

        # Save sampling output to files
        if (output_dir is not None):
            pm.save_trace(trace, output_dir + '/model.trace',
                overwrite=True)
            df = pm.trace_to_dataframe(trace) 
            df.to_csv(output_dir + '/trace.csv',)
            
            # Save output stats to file
            df = pm.summary(trace)
            df = df.round(3)
            df.to_csv(output_dir + '/sampling_stats.csv', sep=' ')

            # Save stats about divergent samples 
            with open(output_dir + "/divergences.txt", "w") as text_file:
                divergent = trace['diverging']
                print(f'Number of Divergent %d' % divergent.nonzero()[0].size,
                        file=text_file)
                divperc = divergent.nonzero()[0].size / len(trace) * 100
                print(f'Percentage of Divergent %.1f' % divperc, file=text_file)

            # Save traceplots
            _ = pm.traceplot(trace)
            plt.savefig(output_dir + '/traceplots.png')

        return trace

    def sample_with_emcee(self, output_dir=None, n_walkers=50, n_samples=10000):
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
            dct = self.bijection.rmap(params)
            args = (dct[k.name] for k in self.vars)
            results = f(*args)
            return tuple(results)

        # First we work out the shapes of all of the deterministic variables
        res = pm.find_MAP()
        vec = self.bijection.map(res)
        initial_blobs = log_prob_func(vec)[1:]
        dtype = [(var.name, float, np.shape(b)) for var,
             b in zip(self.deterministics, initial_blobs)]
        
        # Then sample as usual
        coords = vec + 1e-5 * np.random.randn(n_walkers, len(vec))
        nwalkers, ndim = coords.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func, 
            blobs_dtype=dtype)
        sampler.run_mcmc(coords, n_samples, progress=True)

        # Save sampling output to file
        if (output_dir is not None):
            # Save sampling stats
            with open(output_dir + "/sampling_stats_emcee.txt", "w") as text_file:
                for i, param in enumerate(self.free_parameters):
                    mean = np.mean(sampler.flatchain[:, i])
                    std = np.std(sampler.flatchain[:, i])
                    print(f"parameter, mean, std", file=text_file)
                    print(f"{param}, {mean}, {std}", file=text_file)

            # Save trace to file
            np.save(output_dir + '/trace_emcee.npy', sampler.chain)

            # Save traceplots to file
            fig, ax = plt.subplots(ndim, 1)
            for i, param in enumerate(self.free_parameters):
                mask = sampler.acceptance_fraction > .08
                ax[i].plot(sampler.chain[mask, ::10, i].T, 'k-', alpha=0.2);
                ax[i].set_ylabel(param)

            plt.savefig(output_dir + '/traceplots_emcee.png', bbox_inches='tight')

            # Save corner plot
            figure = corner.corner(sampler.flatchain[2000:, :], 
                quantiles=[0.16, 0.5, 0.84], 
                show_titles=True, labels=self.free_parameters)
            plt.savefig(output_dir + '/corner_emcee.png', bbox_inches='tight')

        return sampler.chain

class OutlierRemovalModel(SingleLensModel):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data):
        super(OutlierRemovalModel, self).__init__(data, 
            errorbar_rescaling='additive_variance', 
            kernel='matern32')

        # Initialize linear parameters
        self.initialize_linear_parameters()
        
        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end,
            testval=self.t0_guess(data))
        self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.05)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=10.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 
       
        # Compute log_likelihood
        self.log_likelihood()

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]
       
    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 

class PointSourcePointLens(SingleLensModel):
    """
    Standard Point Source Point Lens model. 
    """
    def __init__(self, data, errorbar_rescaling='constant', 
            kernel='white_noise'):
        super(PointSourcePointLens, self).__init__(data, errorbar_rescaling, 
            kernel)
        
        # Initialize linear parameters
        self.initialize_linear_parameters() 

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end,
            testval=self.t0_guess(data))
        self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        #self.tE = self.BoundedNormal('tE', mu=0., sd=365., testval=20.)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        # Compute log_likelihood
        self.log_likelihood()

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(T.abs_(self.u0)) - 1) 
    
#    def revert_flux_params_to_nonstandardized_format(self, data):
#        # Revert F_base and Delta_F to non-standardized units
#        median_F = np.median(data.tables[0]['flux'])
#        std_F = np.std(data.tables[0]['flux'])
#
#        Delta_F_ = std_F*self.Delta_F + median_F
#        F_base_ = std_F*self.Delta_F + median_F
#
#        # Calculate source flux and blend flux
#        FS = Delta_F_/(self.peak_mag() - 1)
#        FB = (F_base_ - FS)/FS
#
#        # Convert fluxes to magnitudes
#        mu_m, sig_m = data.fluxes_to_magnitudes(np.array([FS, FB]), 
#            np.array([0., 0.]))
#        mag_source, mag_blend = mu_m
#
#        return mag_source, mag_blend

#    def peak_mag(self):
#        """Returns PSPL magnification at u=u0."""
#        u = T.sqrt(self.u0**2 + ((self.t - self.t0)/self.tE)**2)
#        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
#
#        return A(self.u0)

class FiniteSourcePointLens(SingleLensModel):
    """
    Finite Source Point Lens model. Finite source effects are computed using
    the approach described in http://adsabs.harvard.edu/abs/2004ApJ...603..139Y.
    This approach relies on solving different kinds of elliptic integrals, 
    the solutions for which are provided in an external table FSPL_table.npy.
    This table is loaded once the model is initialized.

    """
    def __init__(self, data, errorbar_rescaling='constant', 
            kernel='white_noise'):
        super(FiniteSourcePointLens, self).__init__(data, errorbar_rescaling, 
            kernel)

        # Initialize linear parameters
        self.initialize_linear_parameters()

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end, 
            testval=self.t0_guess(data))
        self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        # Load FSPL table
        table = np.load('data/FSPL_table.npy')
        points = [table[:, 0]] # z = u/rho_*
        B0 = table[:, 1]
        B1 = table[:, 2]
        B12 = table[:, 3]
        self.B0_interp = xo.interp.RegularGridInterpolator(points, B0[:, None])
        self.B1_interp = xo.interp.RegularGridInterpolator(points, B1[:, None])
#        self.B12_interp = xo.interp.RegularGridInterpolator(points, B12[:, None])

        # Specify finite source parameters
        self.rho_star = self.BoundedNormal('rho_star', mu=0., sd=0.2,
            testval=0.01)
        self.zeta_lambda = self.BoundedNormal('zeta_lambda', 
            mu=T.zeros((self.n_bands, 1)),
            sd=1.*T.ones((self.n_bands, 1)),
            testval=0.5*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))
        
        # Compute the log_likelihood
        self.log_likelihood()

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)

        A = lambda u, z, B0, B1: (u**2 + 2)/(u*T.sqrt(u**2 + 4))*\
            (B0 - self.zeta_lambda*B1)

        z = u/self.rho_star
        
        # Inefficient hacky solution for consturcting B0 and B1 with the same
        # shape as t and u
        B0_list = [] 
        B1_list = []
        for i in range(self.n_bands):
            z_ = z[i].reshape((T.shape(t).eval()[1], 1))
            B0_list.append(self.B0_interp.evaluate(z_).T[0])
            B1_list.append(self.B1_interp.evaluate(z_).T[0])

        B0 = T.stack(B0_list)
        B1 = T.stack(B1_list)
        return (A(u, z, B0, B1) - 1)/(A((T.abs_(self.u0)), z, B0, B1) - 1) 

class PointSourcePointLensAnnualParallax(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant', 
            kernel='white_noise', parametrization='angle_magnitue'):
        super(PointSourcePointLensAnnualParallax, self).__init__(data,
            errorbar_rescaling, kernel)

        self.parametrization = parametrization

        # Initialize linear parameters
        self.initialize_linear_parameters()
        
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end, 
            testval=self.t0_guess(data))

        self.u0 = pm.Normal('u0', mu=0., sd=1.5, testval=-0.41)
#        self.omega_E = self.BoundedNormal('omegaE', mu=0., sd=1., 
#                testval=0.1)
#        self.tE = self.BoundedNormal('tE', mu=0., sd=365., testval=110.)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/T.abs_(self.u0)) 
#        self.tE = pm.Deterministic("tE", 1/self.omega_E) 

        self.initialize_zeta_function_interpolators(data)

        if (parametrization=='angle_magnitude'):
            self.pi_E = self.BoundedNormal('pi_E', mu=0., sd=1., testval=0.1)
            self.psi = xo.distributions.Angle('psi') 

        elif (parametrization=='two_component'):
            self.pi_EE = pm.Normal('pi_EE', mu=0., sigma=1., testval=0.1)
            self.pi_EN = pm.Normal('pi_EN', mu=0., sigma=1., testval=-0.3)
            self.pi_E = pm.Deterministic('pi_E', T.sqrt(self.pi_EE**2 +\
                 self.pi_EN**2))

        else:
            # Acceleration parameters
    #        self.a_par = pm.Exponential('a_par', lam=100, testval=0.001)
    #        self.a_vert = pm.Exponential('a_vert', lam=100, testval=0.001)

            self.a_per = pm.Normal('a_per', mu=0, sd=0.0001, testval=0.0)
            self.a_par = pm.Normal('a_par', mu=0, sd=0.0001, testval=0.0)

            # Save value of pi_E
            self.pi_E = pm.Deterministic('pi_E', 
                T.sqrt((self.a_par**2 + self.a_per**2)/\
                (self.delta_zeta_e_ddot_t0**2 + self.delta_zeta_n_ddot_t0**2)))

        # Compute the log_likelihood
        self.log_likelihood()

        # Store the value of the log-posterior as a deterministic RV
        pm.Deterministic('log_posterior', self.logpt)
    
    def project_vector_onto_sky(self, matrix, coordinates):
        """
        This function takes a 3D cartesian vector specified
        in the ICRS coordinate system and evaluated at differrent
        times (t_i,...t_N) and projects it onto a spherical coordinate
        system on the plane of the sky with the origin at the position
        defined by the coordinates.
        
        Parameters
        ----------
        matrix : ndarray
            Matrix of shape (times_len, 3)
        coordinates : astropy.coordinates.SkyCoord
            Coordinates on the sky.
        """

        # Unit vector normal to the plane of the sky in ICRS coordiantes
        direction = np.array(coordinates.cartesian.xyz.value)
        direction /= np.linalg.norm(direction)
        
        # Unit vector pointing north in ICRS coordinates
        e_north = np.array([0., 0., 1.])

        # Sperical unit vectors of the coordinate system defined
        # on the plane of the sky which is perpendicular to the
        # source star direction
        e_east_sky = np.cross(e_north, direction)
        e_north_sky = np.cross(direction, e_east_sky)

        east_component  = np.dot(matrix, e_east_sky)
        north_component = np.dot(matrix, e_north_sky)
        
        return east_component, north_component

    def initialize_zeta_function_interpolators(self, data):
        """
        This function calls NASA's JPL Horizons to calculate the Earth's 
        orbital elements for each of the observed times. It computes the 
        projection of the Earth--Sun separation vector onto the plane of the 
        sky as well as its derivatives. It then initializes 
        `exoplanet.interp.RegularGridInterpolator` objects as class attributes.
        These interpolators are used to evaluate the zeta function and its
        derivatives at arbitrary times which is needed to compute the 
        likelihood.
        
        Parameters
        ----------
        data : caustic.data
            Caustic data object.
        """
        t = np.linspace(self.t_begin, self.t_end, 5000) + 2450000
        times = Time(t, format='jd', scale='tdb')

        # Get Earth's position and velocity from JPL Horizons
        pos, vel = get_body_barycentric_posvel('earth', times)

        # Minus sign is because get_body_barycentric returns Earth position
        # in heliocentric coordinates 
        s_t = -pos.xyz.value.T
        v_t = -vel.xyz.value.T

        # Acceleration is not provided by the astropy function so we compute
        # the derivative numerically
        a_t = np.gradient(v_t, axis=0)

        # Project vectors onto the plane of the sky
        zeta_e, zeta_n = self.project_vector_onto_sky(s_t, data.coordinates)
        zeta_e_dot, zeta_n_dot = self.project_vector_onto_sky(v_t, 
            data.coordinates)
        zeta_e_ddot, zeta_n_ddot = self.project_vector_onto_sky(a_t, 
            data.coordinates)

        # Define interpolator objects for the zeta function and its derivatives
        points = [t - 2450000]  # Times used for interpolation
        self.zeta_e_interp = xo.interp.RegularGridInterpolator(points, 
            zeta_e[:, None])
        self.zeta_n_interp= xo.interp.RegularGridInterpolator(points, 
            zeta_n[:, None])
        self.zeta_e_dot_interp = xo.interp.RegularGridInterpolator(points, 
            zeta_e_dot[:, None])
        self.zeta_n_dot_interp = xo.interp.RegularGridInterpolator(points, 
            zeta_n_dot[:, None])
        self.zeta_e_ddot_interp = xo.interp.RegularGridInterpolator(points, 
            zeta_e_ddot[:, None])
        self.zeta_n_ddot_interp = xo.interp.RegularGridInterpolator(points, 
            zeta_n_ddot[:, None])

        # Compute 2nd derivatives of zeta functions at t0, this is only 
        # necessary if using the parametrization in terms of local acceleration
        if (self.parametrization=='local_acceleration'):
            self.delta_zeta_e_ddot_t0 = self.zeta_e_ddot_interp.evaluate(\
                    T.reshape(self.t0, (1, 1)))[0, 0]
            self.delta_zeta_n_ddot_t0 = self.zeta_n_ddot_interp.evaluate(\
                    T.reshape(self.t0, (1, 1)))[0, 0]

    def compute_u(self, t, delta_zeta_e, delta_zeta_n):
        """
        Computes the magnitude of the relative lens-source separation vector
        u(t). The parametrization used is specified by the 
        `self.parametrization` property.
        
        Parameters
        ----------
        t : theano.tensor
            Tensor containing times for which u(t) is to be computed. Needs 
            to have shape (n_bands, npoints)
        delta_zeta_e : theano.tensor
            East component of delta_zeta(t). Same shape as t.
        delta_zeta_n : theano.tensor
            East component of delta_zeta(t). Same shape as t.
        
        Returns
        -------
        theano.tensor
            The magnitude of the separation vector u(t) for each time.
        """
        if (self.parametrization=='angle_magnitude'):
            #psi = self.psi + np.pi
            u_per = self.u0 + self.pi_E*T.cos(self.psi)*\
                delta_zeta_e - self.pi_E*T.sin(self.psi)*delta_zeta_n
            u_par = (t - self.t0)*self.omega_E + self.pi_E*T.sin(self.psi)*\
                delta_zeta_e + self.pi_E*T.cos(self.psi)*delta_zeta_n

            return T.sqrt(u_par**2 + u_per**2)
        
        elif (self.parametrization=='local_acceleration'):
            # Compute u(t)
            piE_cospsi = (self.a_par*self.delta_zeta_n_ddot_t0 +\
                self.a_per*self.delta_zeta_e_ddot_t0)/(self.delta_zeta_e_ddot_t0**2 +\
                    self.delta_zeta_n_ddot_t0**2)
            piE_sinpsi = (self.a_par*self.delta_zeta_e_ddot_t0 -\
                self.a_per*self.delta_zeta_n_ddot_t0)/(self.delta_zeta_e_ddot_t0**2 +\
                    self.delta_zeta_n_ddot_t0**2)

            u_per = self.u0 + piE_cospsi*\
                delta_zeta_e - piE_sinpsi*delta_zeta_n
            u_par = (t - self.t0)/self.tE + piE_sinpsi*\
                delta_zeta_e + piE_cospsi*delta_zeta_n

            return T.sqrt(u_par**2 + u_per**2)  

        else: 
            u_per = self.u0 + self.pi_EN*delta_zeta_e - self.pi_EE*delta_zeta_n
            u_par = (t - self.t0)/self.tE + self.pi_EE*delta_zeta_e +\
                self.pi_EN*delta_zeta_n

            return T.sqrt(u_par**2 + u_per**2)

    def compute_delta_zeta(self, t):
        """
        Computes the components of delta_zeta vector - the devaition of the 
        projected separation of the Sun relative to time t0.
        """
        zeta_e_list = []
        zeta_n_list = []

        for idx in range(self.n_bands):
            # the interpolation function requires tensor of shape (n_points, 1)
            # as an input 
            pts = T.reshape(t[idx], (T.shape(t[idx])[0], 1))
            zeta_e_list.append(self.zeta_e_interp.evaluate(pts).transpose())
            zeta_n_list.append(self.zeta_n_interp.evaluate(pts).transpose())

        # Stack interpolated functions such that they have the same shape as
        # self.t, namely, (n_bands, npoints)
        zeta_e = T.concatenate(zeta_e_list, axis=1)
        zeta_n = T.concatenate(zeta_n_list, axis=1)

        zeta_e_t0 = self.zeta_e_interp.evaluate(T.reshape(self.t0,
             (1, 1)))[0, 0]
        zeta_n_t0 = self.zeta_n_interp.evaluate(T.reshape(self.t0,
             (1, 1)))[0, 0]
        zeta_e_dot_t0 = self.zeta_e_dot_interp.evaluate(T.reshape(self.t0,
            (1, 1)))[0, 0]
        zeta_n_dot_t0 = self.zeta_n_dot_interp.evaluate(T.reshape(self.t0,
            (1, 1)))[0, 0]

        # Compute delta_zeta function 
        delta_zeta_e = zeta_e - zeta_e_t0 -\
             (t - self.t0)*zeta_e_dot_t0
        delta_zeta_n = zeta_n - zeta_n_t0 -\
             (t - self.t0)*zeta_n_dot_t0

        return delta_zeta_n, delta_zeta_e

    def magnification(self, t):
        """
        Computes the magnification fraction [A(u) - 1]/[A(u0) - 1]
        where A(u) is the magnification function.
        
        Parameters
        ----------
        t : theano.tensor
            Tensor containing times for which A(t) is to be computed. Needs 
            to have shape (n_bands, max_npoints)
        
        Returns
        -------
        theano.tensor
            The value of the magnification at each time t.
        """
        delta_zeta_n, delta_zeta_e = self.compute_delta_zeta(t)

        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
        u = self.compute_u(t, delta_zeta_e, delta_zeta_n)

        return (A(u) - 1)/(A(T.abs_(self.u0)) - 1) 

class PointSourcePointLensMarginalized(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant'):
        super(PointSourcePointLensMarginalized, self).__init__(data)
        
        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end,
            testval=self.t0_guess(data))
        self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        # Compute the likelihood function
        self.mag = self.magnification(self.t) 

        self.compute_varF()

        # Compute the log-likelihood which is additive across different bands
        ll = 0 
        for i in range(self.n_bands):
            ll += self.log_likelihood_single_band(self.F[i], 
                self.varF[i], self.mag[i], self.mask[i])

        pm.Potential('log_likelihood', ll)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]
        
    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 
    
    def log_likelihood_single_band(self, F, varF, mag, mask):
        F = F[mask.nonzero()]
        varF = varF[mask.nonzero()]
        mag = mag[mask.nonzero()]

        N = T.shape(F)[0]
        # Linear parameter matrix
#        mu_theta = T.dot(mag_vector, np.max(self.F))
        A = T.stack([mag, T.ones_like(F)], axis=1)

        # Covariance matrix
        C_diag = varF
        C = T.nlinalg.diag(C_diag)

        # Prior matrix
        sigDelta_F = 10.
        sigF_base = 0.1
        L_diag = T._shared(np.array([sigDelta_F, sigF_base])**2.)
        L = T.nlinalg.diag(L_diag)

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
        return -0.5*T.dot(r.transpose(), T.dot(inv_SIGMA, r)) -\
            0.5*N*np.log(2*np.pi) - 0.5*ln_detSIGMA