import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from matplotlib import pyplot as plt

import exoplanet as xo
from exoplanet.gp import terms, GP

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

from astroquery.jplhorizons import Horizons
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
            'constant', 'additive_variance', and  'flux_dependant', by default 
            'constant'.
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
        Calculates the PSPL magnification fraction [A(u) - 1]/[A(u0) - 1]
        where A(u) is the analytic PSPL magnification.
        """
        pass

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
        if (self.errorbar_rescaling=='constant'):
            ## Noise model parameters
            self.A = self.BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    testval=1.5*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            
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

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            self.logp_B = pm.Deterministic('logp_B',
                pm.Normal.dist(
                    mu=T.zeros((self.n_bands, 1)), 
                    sd=1*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.B))

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

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            self.logp_B = pm.Deterministic('logp_B',
                pm.Normal.dist(
                    mu=T.zeros((self.n_bands, 1)), 
                    sd=5*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.B))

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
            Returns parameters of an inverse gamma distribution p(x) such that 
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

        # Save log prior for each parameter, this is needed for hierarchical
        # modeling of multiple events using the importance resampling trick
        self.logp_sigma = pm.Deterministic('logp_sigma',
            self.BoundedNormal.dist(
                mu=T.zeros((self.n_bands, 1)),
                sd=3.*T.ones((self.n_bands, 1)),
                testval=0.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.sigma))
        self.logp_rho = pm.Deterministic('logp_rho',
            pm.InverseGamma.dist(
                alpha = T._shared(invgamma_a)*T.ones((self.n_bands, 1)),
                beta = T._shared(invgamma_b)*T.ones((self.n_bands, 1)),
                testval=0.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.rho))

    def log_likelihood(self):
        """"
        Computes the total log likelihood of the model assuming that the 
        observations in different bands are independent. The choice of 
        the likelihood function is specified when initializing the model.
        """
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
        mask_numpy  = mask.nonzero()[0].eval().astype(bool)
        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[:, i, ~mask_numpy])

        return pred_list

    def evaluate_prior_model_on_grid(self, trace, t_grids):
        """
        Evaluates a posterior mean model on a dense grid given samples from a 
        posterior distribution over the model parameters.
        
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
        mask_numpy  = mask.nonzero()[0].eval().astype(bool)
        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[:, i, ~mask_numpy])

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

                print(type(gp.predict(t_grids[n])))
                print(np.shape(t_grids[n]))
                T.printing.Print('TEST')(gp.predict(t_grids[n]))

                prediction_eval[n] =\
                    xo.eval_in_model(gp.predict(t_grids[n]), map_point) 

            # Add mean model to GP prediction
            prediction_eval += xo.eval_in_model(prediction_mean, map_point)

        # Construct list of prediction_eval
        mask_numpy  = mask.nonzero()[0].eval().astype(bool)
        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[i, ~mask_numpy])

        return pred_list

    def sample(self, output_dir, n_tune=2000, n_samples=2000, 
            target_accept=0.9):
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
            various diagnostic plots and information.
        n_tune : int, optional
            Number of tuning steps, by default 2000
        n_samples : int, optional
            Number of sampling steps, by default 2000
        target_accept : float, optional
            Target acceptance fraction for the NUTS sampler, high
            The step size for the leapfrog integrator in NUTS is tuned such 
            to approximate this acceptance rate. Higher values like 0.9 or 
            0.95 often work better for problematic posteriors, by default 0.9.
        """

        sampler = xo.PyMC3Sampler(window=100, start=200, finish=200)

        print("Free parameters:\n", self.free_parameters)
        print("Initial values of logp for each parameter:\n", 
            self.initial_logps)

        # Run burn-in chains             
        burnin = sampler.tune(tune=n_tune,
                step_kwargs=dict(target_accept=target_accept))

        # Run final sampling
        trace = sampler.sample(draws=n_samples,
                step_kwargs=dict(target_accept=target_accept))

        # Save trace as a MultiTrace object and a csv
        pm.save_trace(trace, output_dir + '/model.trace',
            overwrite=True)
        df = pm.trace_to_dataframe(trace) 
        df.to_csv(output_dir + '/trace.csv',)
        
        # Save output stats to file
        df = pm.summary(trace)
        df = df.round(3)
        df.to_csv(output_dir + '/sampling_stats.csv')

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

    def sample_with_emcee(self, output_dir, n_walkers=50, n_samples=10000):
        """
        Samples the model posterior distribution using emcee.
                
        Parameters
        ----------
        output_dir : str
            Path to directory where the trace is to be saved, together with
            various diagnostic plots and information.
        n_walkers: int, optional
            Number of walkers, by default 50.
        n_samples : int, optional
            Number of sampling steps, by default 10000.
        """
        print("Free parameters:\n", self.free_parameters)
        print("Initial values of logp for each parameter:\n", 
            self.initial_logps)

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
        self.Delta_F = self.BoundedNormal('Delta_F', 
            mu=T.zeros((self.n_bands, 1)),
            sd=T.max(self.F)*T.ones((self.n_bands, 1)),
            testval=T.max(self.F)*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        self.F_base = pm.Normal('F_base', 
            mu=T.zeros((self.n_bands, 1)), 
            sd=3.*T.ones((self.n_bands, 1)),
            testval=T.zeros((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end,
            testval=self.t0_guess(data))
        self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.05)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=10.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 
       
        # Compute the mean model and the residuals
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base 
        self.r = self.F - mean_func

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

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end,
            testval=self.t0_guess(data))
        self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        ## Save log prior for each parameter for hierarhical modeling 
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            self.BoundedNormal.dist(
                mu=T.zeros((self.n_bands, 1)),
                sd=50.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t_begin, self.t_end).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            self.BoundedNormal.dist(mu=0., sd=1.5).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            self.BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))

        # Compute the mean model and the residuals
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base 
        self.r = self.F - mean_func

        # Compute log_likelihood
        self.log_likelihood()

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 
    
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

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', self.t_begin, self.t_end, 
            testval=self.t0_guess(data))
        self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = self.BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        ## Save log prior for each parameter for hierarhical modeling 
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            self.BoundedNormal.dist(
                mu=T.zeros((self.n_bands, 1)),
                sd=50.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t_begin, self.t_end).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            self.BoundedNormal.dist(mu=0., sd=1.5).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            self.BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))

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
        self.gamma_lambda = self.BoundedNormal('gamma_lambda', 
            mu=T.zeros((self.n_bands, 1)),
            sd=1.*T.ones((self.n_bands, 1)),
            testval=0.5*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))
        
        # Compute the mean function and the residuals
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base 
        self.r = self.F - mean_func

        # Compute the log_likelihood
        self.log_likelihood()

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)

        A = lambda u, z, B0, B1: (u**2 + 2)/(u*T.sqrt(u**2 + 4))*\
            (B0 - self.gamma_lambda*B1)

        z = u/self.rho_star

        # This is incredibly inefficient
        B0_list = [] 
        B1_list = []
        for i in range(self.n_bands):
            z_ = z[i].reshape((self.max_npoints, 1))
            B0_list.append(self.B0_interp.evaluate(z_).T[0])
            B1_list.append(self.B1_interp.evaluate(z_).T[0])

        B0 = T.stack(B0_list)
        B1 = T.stack(B1_list)
        return (A(u, z, B0, B1) - 1)/(A(self.u0, z, B0, B1) - 1) 

class PointSourcePointLensAnnualParallax(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant', 
            kernel='white_noise', parametrization='angle_magnitue'):
        super(PointSourcePointLensAnnualParallax, self).__init__(data,
            errorbar_rescaling, kernel)

        # Load orbital data from JPL Horizons
        start = Time(data.tables[0]['HJD'][0], format='jd')
        stop = Time(data.tables[0]['HJD'][-1], format='jd')

        epochs={'start':start.iso[:10], 'stop':stop.iso[:10], 'step':'6h'}
        obj = Horizons(id='399', id_type='id', epochs=epochs)

        elements = obj.elements()

        # Calculate the projection of Eath-Sun position vector on the plane
        # of the sky
        coordinates_ecliptic = \
            data.coordinates.transform_to('geocentrictrueecliptic')
        lambda_0  = coordinates_ecliptic.lon.value*(np.pi/180)
        beta_0  = coordinates_ecliptic.lat.value*(np.pi/180)

        t = np.array(elements['datetime_jd']) # JD
        e = np.array(elements['e'])
        tp = np.array(elements['Tp_jd']) # JD
        n = (2*np.pi/365.25) # mean motion
        Phi_gamma = (77.86)*np.pi/180 # true anomaly at vernal eq. on J2000

        r_sun = 1 - e*np.cos(n*(t - tp)) # to 1st order in e
        lambda_sun = n*(t - tp) - Phi_gamma + 2*e*np.sin(n*(t - tp)) # to 1st order in e

        gamma_w = r_sun*np.sin((lambda_sun - lambda_0))
        gamma_n = r_sun*np.sin(beta_0)*np.cos(lambda_sun - lambda_0)

        # Need to define an interpolator for all gamma functions for
        points = [t] # Times for which we have JPL Horizons data
        self.gamma_w_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_w[:, None])
        self.gamma_n_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_n[:, None])

        # Interpolate gamma functions onto the grid of observed times, this 
        # happens only once when the model is initialized
        n_max =  self.max_npoints
        gamma_w = T._shared(np.stack([np.pad(
            self.gamma_w_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))
        gamma_n = T._shared(np.stack([np.pad(
            self.gamma_n_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))

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

        if (parametrization=='angle_magnitude'):
            self.t0 = pm.Uniform('t0', self.t_begin, self.t_end, 
                testval=self.t0_guess(data))
            self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
            self.omega_E = self.BoundedNormal('omega_E', mu=0., sd=1., 
                testval=0.1)
            self.pi_E = self.BoundedNormal('pi_E', mu=0., sd=1., testval=0.01)
            self.psi = xo.distributions.Angle('psi') 

            mag = self.magnification(self.t, gamma_w, gamma_n, parametrization) 
            
        elif (parametrization=='two_component'):
            self.t0 = pm.Uniform('t0', self.t_begin, self.t_end, 
                testval=self.t0_guess(data))
            self.u0 = self.BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
            self.omega_E = self.BoundedNormal('omega_E', mu=0., sd=1., 
                testval=0.1)
            self.pi_EW = pm.Normal('pi_EW', mu=0., sigma=1., testval=0.01)
            self.pi_EN = pm.Normal('pi_EN', mu=0., sigma=1., testval=0.01)

            mag = self.magnification(self.t, gamma_w, gamma_n, parametrization) 

        else:
            # Compute 1st and 2nd order derivatives of the gamma vector
            gamma_w_dot = n*((1 + e*np.cos(n*(t - tp)))*\
                np.cos(lambda_sun - lambda_0)+\
                e*np.sin(n*(t - tp))*np.sin(lambda_sun - lambda_0))
            gamma_n_dot = -n*np.sin(beta_0)*((1 + e*np.cos(n*(t - tp)))\
                *np.sin(lambda_sun - lambda_0) -\
                e*np.sin(n*(t - tp))*np.cos(lambda_sun - lambda_0))

            gamma_w_ddot = -n**2*(1 + 2*e*np.cos(n*(t - tp)))*\
                np.sin(lambda_sun - lambda_0)
            gamma_n_ddot = -n**2*np.sin(beta_0)*(1 + 2*e*np.cos(n*(t - tp)))*\
                np.cos(lambda_sun - lambda_0)

            # Initialize interpolators
            self.gamma_w_dot_interp = xo.interp.RegularGridInterpolator(points, 
                gamma_w_dot[:, None])
            self.gamma_n_dot_interp = xo.interp.RegularGridInterpolator(points, 
                gamma_n_dot[:, None])
            self.gamma_w_ddot_interp = xo.interp.RegularGridInterpolator(points, 
                gamma_w_ddot[:, None])
            self.gamma_n_ddot_interp = xo.interp.RegularGridInterpolator(points, 
                gamma_n_ddot[:, None])

            # Acceleration parameters
    #        self.a_par = pm.Exponential('a_par', lam=10, testval=0.01)
    #        self.a_vert = pm.Exponential('a_vert', lam=10, testval=0.01)

            self.a_par = pm.Normal('a_par', mu=0, sd=0.1, testval=0.01)
            self.kappa_0 = pm.Normal('kappa_0', mu=0, sd=0.1, testval=0.01)
            self.t0_prime = pm.Uniform('t0_prime', self.t_begin, self.t_end,
                testval=self.t0_guess(data))
            self.u0_prime = pm.Normal('u0_prime', mu=0., sd=1., testval=0.05)
            self.v0_prime = pm.Normal('v0_prime', mu=0., sd=1., testval=0.05)

            # Compute the magnification
            self.t_E = 0.
            self.pi_E = 0.
            mag = self.magnification_3(self.t, gamma_w, gamma_n) 

            # Save t_E, and pi_E parameters to trace
            pm.Deterministic('t_E', self.t_E)
            pm.Deterministic('pi_E', self.pi_E)

        # Compute the mean function and the residuals
        mean_func = self.Delta_F*mag +  self.F_base 
        self.r = self.F - mean_func

        # Compute the log_likelihood
        self.log_likelihood()

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

    def compute_u_1(self, t, gamma_w, gamma_n):
        psi = self.psi + np.pi
        u_w = -self.u0*T.sin(psi) - self.omega_E*(t - self.t0)*T.cos(psi) +\
            self.pi_E*gamma_w
        u_n = self.u0*T.cos(psi) - self.omega_E*(t - self.t0)*T.sin(psi) +\
            self.pi_E*gamma_n

        return T.sqrt(u_w**2 + u_n**2)

    def compute_u_2(self, t, gamma_w, gamma_n):
        u_w = -self.omega_E*(t - self.t0) + self.pi_EW*gamma_w +\
             self.pi_EN*gamma_n
        u_n = self.u0 + self.pi_EW*gamma_n - self.pi_EN*gamma_w

        return T.sqrt(u_w**2 + u_n**2)

    def compute_u_3(self, t, gamma_w, gamma_n):
        # Parallax parameters
        gamma_w_t0 = self.gamma_w_interp.evaluate(\
            T.reshape(self.t0_prime + 2450000, (1, 1)))[0, 0] 
        gamma_n_t0 = self.gamma_n_interp.evaluate(\
            T.reshape(self.t0_prime + 2450000, (1, 1)))[0, 0]
        gamma_w_dot_t0 = self.gamma_w_dot_interp.evaluate(\
            T.reshape(self.t0_prime + 2450000, (1, 1)))[0, 0]
        gamma_n_dot_t0 = self.gamma_n_dot_interp.evaluate(\
            T.reshape(self.t0_prime + 2450000, (1, 1)))[0, 0]
        gamma_w_ddot_t0 = self.gamma_w_ddot_interp.evaluate(\
            T.reshape(self.t0_prime + 2450000, (1, 1)))[0, 0]
        gamma_n_ddot_t0 = self.gamma_n_ddot_interp.evaluate(\
            T.reshape(self.t0_prime + 2450000, (1, 1)))[0, 0]

        a_vert = self.kappa_0 - self.v0_prime**2/self.u0_prime**2

        # Compute u(t)
        pi_E = T.sqrt((self.a_par**2 +\
             a_vert**2)/(gamma_w_ddot_t0**2 + gamma_n_ddot_t0**2))

        sin_phi = pi_E*(a_vert*gamma_w_ddot_t0 -\
             self.a_par*gamma_n_ddot_t0)/(self.a_par**2 + a_vert**2)

        cos_phi = pi_E*(a_vert*gamma_n_ddot_t0 +\
             self.a_par*gamma_w_ddot_t0)/(self.a_par**2 + a_vert**2)

        u_w = self.u0_prime*sin_phi + pi_E*(gamma_w - gamma_w_t0) +\
            (self.v0_prime*cos_phi - pi_E*gamma_w_dot_t0)*(t - self.t0_prime)
        u_n = self.u0_prime*cos_phi + pi_E*(gamma_n - gamma_n_t0) +\
            (-self.v0_prime*sin_phi - pi_E*gamma_n_dot_t0)*(t - self.t0_prime)

        # Deterministic transform
        t_E =  1/T.sqrt(self.v0_prime*2 -\
            2*self.v0_prime*pi_E*(gamma_w_dot_t0*cos_phi - gamma_n_dot_t0*sin_phi)\
                + pi_E**2*(gamma_w_dot_t0**2 + gamma_n_dot_t0**2))
        
        self.t_E = t_E
        self.pi_E = pi_E

        return T.sqrt(u_w**2 + u_n**2)  

    def magnification(self, t, gamma_w, gamma_n, parametrization):
        if (parametrization=='angle_magnitude'):
            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
            u = self.compute_u_1(t, gamma_w, gamma_n)
            return (A(u) - 1)/(A(self.u0) - 1) 

        elif (parametrization=='two_component'):
            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
            u = self.compute_u_2(t, gamma_w, gamma_n)
            return (A(u) - 1)/(A(self.u0) - 1) 

        else:
            A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
            u = self.compute_u_3(t, gamma_w, gamma_n)
            return (A(u) - 1)/(A(self.u0_prime) - 1) 
    
    def evaluate_posterior_model_on_grid(self, trace, t_grids, n_samples=50):
        n_max =  np.max([len(t_grid) for t_grid in t_grids])
        prediction_eval = np.zeros((n_samples, self.n_bands, n_max))

        # Construct tensors needed to evaluate the model 
        t_grids_tensor, mask = self.construct_masked_tensor(t_grids)

        gamma_w, _ = self.construct_masked_tensor(
            [self.gamma_w_interp.evaluate(t_grid[:, None] + 2450000).eval().ravel()\
                for t_grid in t_grids])

        gamma_n, _ = self.construct_masked_tensor(
            [self.gamma_n_interp.evaluate(t_grid[:, None] + 2450000).eval().ravel()\
                for t_grid in t_grids])

        # Evaluate model for each sample
        for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                size=n_samples)):
            # Tensor which is to be evaluated in model context
            prediction = self.Delta_F*self.magnification(t_grids_tensor, 
                gamma_w, gamma_n) + self.F_base

            prediction_eval[i] = xo.eval_in_model(prediction, sample)

        # Construct list of prediction_eval
        mask_numpy  = mask.nonzero()[0].eval().astype(bool)
        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[:, i, ~mask_numpy])

        return pred_list

    def evaluate_prior_model_on_grid(self, trace, t_grids):
        n_samples = len(np.atleast_1d(trace['t0_prime']))
        n_max =  np.max([len(t_grid) for t_grid in t_grids])
        prediction_eval = np.zeros((n_samples, self.n_bands, n_max))

        # Construct tensors needed to evaluate the model 
        t_grids_tensor, mask = self.construct_masked_tensor(t_grids)

        gamma_w, _ = self.construct_masked_tensor(
            [self.gamma_w_interp.evaluate(t_grid[:, None] + 2450000).eval().ravel()\
                for t_grid in t_grids])

        gamma_n, _ = self.construct_masked_tensor(
            [self.gamma_n_interp.evaluate(t_grid[:, None] + 2450000).eval().ravel()\
                for t_grid in t_grids])

        # Evaluate model for each sample
        for i in range(n_samples):
            # Tensor which is to be evaluated in model context
            prediction = self.Delta_F*self.magnification(t_grids_tensor, 
                gamma_w, gamma_n) + self.F_base

            if (n_samples==1):
                sample = {key:trace[key] for key in trace.keys()}
            else:
                sample = {key:trace[key][i] for key in trace.keys()}

            prediction_eval[i] = xo.eval_in_model(prediction, sample)

        # Construct list of prediction_eval
        mask_numpy  = mask.nonzero()[0].eval().astype(bool)
        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[:, i, ~mask_numpy])

        return pred_list

    def evaluate_map_model_on_grid(self, t_grids, map_point):
        n_max =  np.max([len(t_grid) for t_grid in t_grids])
        prediction_eval = np.zeros((self.n_bands, n_max))

        # Construct tensors needed to evaluate the model
        t_grids_tensor, mask = self.construct_masked_tensor(t_grids)

        gamma_w, _ = self.construct_masked_tensor(
            [self.gamma_w_interp.evaluate(t_grid[:, None] + 2450000).eval().ravel()\
                for t_grid in t_grids])

        gamma_n, _ = self.construct_masked_tensor(
            [self.gamma_n_interp.evaluate(t_grid[:, None] + 2450000).eval().ravel()\
                for t_grid in t_grids])

 
        # Tensor which is to be evaluated in model context
        prediction = self.Delta_F*self.magnification(t_grids_tensor, gamma_w,
            gamma_n) + self.F_base
            
        prediction_eval = xo.eval_in_model(prediction, map_point)

        # Construct list of prediction_eval
        mask_numpy  = mask.nonzero()[0].eval().astype(bool)
        pred_list = []
        for i in range(self.n_bands):
            pred_list.append(prediction_eval[i, ~mask_numpy])

        return pred_list

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

        ## Save log prior for each parameter for hierarhical modeling 
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(self.t_begin, self.t_end).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            self.BoundedNormal.dist(mu=0., sd=1.5).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            self.BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))

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