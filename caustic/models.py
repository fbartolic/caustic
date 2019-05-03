import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from matplotlib import pyplot as plt

import exoplanet as xo
from exoplanet.gp import terms, GP

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy import units as u

class SingleLensModel(pm.Model):
    """
    Skeleton class for a single lens model. Classes which inherit from this class 
    should implement the `magnification`, `log_likelihood`, 
    `evaluate_posterior_model_on_grid` and `evaluate_map_model_on_grid` methods.
    """
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, name='', model=None):
        super(SingleLensModel, self).__init__(name, model)

        # Load and rescale the data to zero median and unit variance
        tables = data.get_standardized_data()
        self.n_bands = len(tables) # number of photometric bands

        ## To avoid loops, we pad the arrays with additional values, in
        ## particular, we padd the flux arrays with very large values such
        ## the likelihood for those points is zero. The final shape of the
        ## arrays is (self.n_bands, n_datapoints) and we can iterate over the bands
        n_max =  np.max([len(table) for table in tables])
        self.t = T._shared(np.stack([np.pad(table['HJD'], 
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in tables]))
        self.F = T._shared(np.stack([np.pad(table['flux'], 
            (0, n_max - len(table['flux'])), 'constant',
            constant_values=(0.,)) for table in tables]))
        self.sigF = T._shared(np.stack([np.pad(table['flux_err'], 
            (0, n_max - len(table['flux_err'])), 'constant',
            constant_values=(0.,)) for table in tables]))

        # Masking array which is later used to mask out the padded values
        masks_list = []
        for table in tables:
            array = np.append(
                np.ones(len(table['HJD'])),  
                np.zeros(n_max - len(table['HJD']))
                )
            masks_list.append(array)

        self.mask = T._shared(np.stack(masks_list).astype('int8'))

    def magnification(self, t):
        """
        Calculates the PSPL magnification fraction [A(u) - 1]/[A(u0) - 1]
        where A(u) is the analytic PSPL magnification.
        """
        pass

    def log_likelihood_single_band(self):
        """
        Implements a white noise Gaussian likelihood function, assuming
        that the observations in each photometric band are independent. 
        Subclasses should overload this method if necessary.
        """
        pass

   
    def t0_guess(self, event):
        """
        Guesses an intial value for the t0 parameter. This is necessary because
        the posterior is highly multi-modal in t0 and the sampler takes
        ages to converge if t0 is not close to true value.
        """
        tmp = event.masks 
        event.remove_worst_outliers(window_size=30, mad_cutoff=2)
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
    
    def evaluate_posterior_model_on_grid(self, trace, t_grids, n_samples=50):
        """
        Evaluates a posterior model on a dense grid given samples from a 
        posterior distribution over the model parameters.
        
        Parameters
        ----------
        trace : PyMC3 MultiTrace object
            Trace object containing samples from posterior.
        t_grids : list of 1D Theano tensors
            Time grids on which the model is to be evaluated. The list should
            contain one tensor for each observing band.
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
        pass

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
        t_grids : list of 1D Theano tensors
            Time grids on which the model is to be evaluated. The list should
            contain one tensor for each observing band.
        
        Returns
        -------
        list
            List of numpy arrays of shape (n_sample, len(t_grid)) for
            each t_grid in t_grids. Each element of the list corresponds to 
            a model prediction in a different band.
        """
        pass

    def evaluate_map_model_on_grid(self, t_grids, map_point):
        """
        Evaluates a MAP model on a dense grid given the MAP parameters of
        the model.
        
        Parameters
        ----------
        t_grids : list of 1D Theano tensors
            Time grids on which the model is to be evaluated. The list should
            contain one tensor for each observing band.
        map_point : dict
            A dictionary contatining values of MAP parameters.

        Returns
        -------
        list
            List of numpy arrays of shape (len(t_grid)) for
            each t_grid in t_grids. Each element of the list corresponds to 
            a model MAP prediction in a different band.
        """
        pass

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
#        _ = pm.traceplot(trace)
#        plt.savefig(output_dir + '/traceplots.png')

        # Save autocorrelation plots for the chains
        pm.plots.autocorrplot(trace)
        plt.savefig(output_dir + '/autocorr.png')

        # Save corner plot of the samples
#        rvs = [rv.name for rv in self.basic_RVs]
#        pm.pairplot(trace,
#                    divergences=True, plot_transformed=True, text_size=25,
#                    varnames=rvs,
#                    color='C3', figsize=(40, 40), 
#                    kwargs_divergence={'color': 'C0'})
#        plt.savefig(output_dir + '/pairplot.png')

class OutlierRemovalModel(SingleLensModel):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data):
        super(OutlierRemovalModel, self).__init__(data)

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize linear parameters
        self.Delta_F = BoundedNormal('Delta_F', 
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
        self.t0 = pm.Uniform('t0', T.min(self.t[0][self.mask[0].nonzero()]), 
            T.max(self.t[0][self.mask[0].nonzero()]), 
            testval=self.t0_guess(data))
        self.u0 = BoundedNormal('u0', mu=0., sd=1.5, testval=0.05)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=10.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        ## Noise model parameters
        self.A = BoundedNormal1('A', 
            mu=T.ones((self.n_bands, 1)),
            sd=2.*T.ones((self.n_bands, 1)),
            testval=1.5*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        self.B = BoundedNormal('B', 
            mu=T.zeros((self.n_bands, 1)), 
            sd=1*T.ones((self.n_bands, 1)),
            testval=0.01*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        ## Compute parameters for the prior on GP hyperparameters
        tables = data.get_standardized_data()
        invgamma_a = np.zeros(self.n_bands)
        invgamma_b = np.zeros(self.n_bands)

        for i in range(len(tables)):
            t = np.array(tables[i]['HJD'])
            invgamma_a[i], invgamma_b[i] = fsolve(self.solve_for_invgamma_params, 
                (0.1, 0.1), (np.median(np.diff(t)), t[-1] - t[0]))

        self.sigma = BoundedNormal('sigma', 
            mu=T.zeros((self.n_bands, 1)),
            sd=3.*T.ones((self.n_bands, 1)),
            testval=0.5*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        self.rho = pm.InverseGamma('rho', 
            alpha = T._shared(invgamma_a)*T.ones((self.n_bands, 1)),
            beta = T._shared(invgamma_b)*T.ones((self.n_bands, 1)),
            testval=2.*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))
        
        # Compute the likelihood function
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base # mean function

        # Residuals
        self.r = self.F - mean_func

        # Diagonal terms of the covariance matrix
        self.varF = T.pow(self.A*self.sigF, 2) + T.pow(self.B, 2)

        # Compute the log-likelihood which is additive across different bands
        ll = 0 
        for i in range(self.n_bands):
            ll += self.log_likelihood_single_band(self.t[i], self.r[i], 
                self.varF[i], self.sigma[i], self.rho[i], self.mask[i])

        pm.Potential('log_likelihood', ll)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]
       
    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 
    
    def solve_for_invgamma_params(self, params, x_min, x_max):
        """
        Returns parameters of an inverse gamma distribution p(x) such that 
        0.1% of total prob. mass is assigned to values of x < x_min and 
        1% of total prob. masss  to values greater than x_max.
        """

        def inverse_gamma_cdf(x, alpha, beta):
            return invgamma.cdf(x, alpha, scale=beta)

        lower_mass = 0.01
        upper_mass = 0.99

        # Trial parameters
        alpha, beta = params

        # Equation for the roots defining params which satisfy the constraint
        return (inverse_gamma_cdf(x_min, alpha, beta) - \
            lower_mass, inverse_gamma_cdf(x_max, alpha, beta) - upper_mass)

    def log_likelihood_single_band(self, t, r, varF, sigma, rho, mask):
        # Calculate likelihood
        kernel = terms.Matern32Term(sigma=sigma, rho=rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        gp = GP(kernel, t[mask.nonzero()], varF[mask.nonzero()], J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the 
        # GP likelihood
        return gp.log_likelihood(r[mask.nonzero()])
   
    def evaluate_map_model_on_grid(self, t_grids, map_point):
        predictions = [np.zeros(T.shape(t_grid).eval()) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate mean function at observed times
            mag_obs = self.magnification(self.t[n][self.mask[n].nonzero()])
            mean_func_obs = self.Delta_F[n]*mag_obs + self.F_base[n]

            # Evaluate mean function on fine grid
            mag = self.magnification(t_grids[n]) 
            mean_func = self.Delta_F[n]*mag + self.F_base[n]
            
            # Diagonal terms of the covariance matrix
            varF = T.pow(self.A[n]*self.sigF[n][self.mask[n].nonzero()], 2) +\
                    T.pow(self.B[n], 2)

            # Initialize kernel and gp object
            kernel = terms.Matern32Term(sigma=self.sigma[n], 
                rho=self.rho[n])
            gp = GP(kernel, self.t[n][self.mask[n].nonzero()], varF, J=2)

            # Calculate log_likelihood 
            r = self.F[n][self.mask[n].nonzero()] - mean_func_obs
            gp.log_likelihood(r)

            # Evaluate tensors in model context
            predictions[n] =\
                xo.eval_in_model(gp.predict(t_grids[n]), map_point) +\
                xo.eval_in_model(mean_func, map_point) 
        
        return predictions 

class PointSourcePointLens(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant'):
        super(PointSourcePointLens, self).__init__(data)

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize linear parameters
        self.Delta_F = BoundedNormal('Delta_F', 
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
        self.t0 = pm.Uniform('t0', T.min(self.t[0][self.mask[0].nonzero()]), 
            T.max(self.t[0][self.mask[0].nonzero()]), 
            testval=self.t0_guess(data))
        self.u0 = BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        ## Save log prior for each parameter for hierarhical modeling 
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            BoundedNormal.dist(
                mu=T.zeros((self.n_bands, 1)),
                sd=50.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(
                T.min(self.t[0][self.mask[0].nonzero()]), 
                T.max(self.t[0][self.mask[0].nonzero()])).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            BoundedNormal.dist(mu=0., sd=1.5).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))

        # Compute the likelihood function
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base # mean function

        # Residuals
        self.r = self.F - mean_func

        if (errorbar_rescaling=='constant'):
            # Define custom prior distributions 
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
            BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

            ## Noise model parameters
            self.A = BoundedNormal1('A', 
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
            self.varF = T.pow(self.A*self.sigF, 2) 

        if (errorbar_rescaling=='additive_variance'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
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
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(self.B, 2)

        if (errorbar_rescaling=='flux_dependant'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
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
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(mag*self.B, 2)

        # Compute the log-likelihood which is additive across different bands
        ll = 0 
        for i in range(self.n_bands):
            ll += self.log_likelihood_single_band(self.r[i], 
                self.varF[i], mag[i], self.mask[i])

        pm.Potential('log_likelihood', ll)

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

    def log_likelihood_single_band(self, r, varF, mag, mask):
        """
        Implements a white noise Gaussian likelihood function, assuming
        that the observations in each photometric band are independent. 
        """
        # Gaussian likelihood
        ll = -0.5*T.sum(T.pow(r[mask.nonzero()], 2.)/varF[mask.nonzero()]) -\
                0.5*T.sum(T.log(2*np.pi*varF[mask.nonzero()]))

        return ll
        
    def evaluate_posterior_model_on_grid(self, trace, t_grids, n_samples=50):
        # List of numpy arrays with shape(n_samples, n_datapoints) array with 
        # model predictions
        predictions = [np.zeros((n_samples, 
            int(T.shape(t_grid).eval()))) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate model for each sample
            for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                    size=n_samples)):
                # Tensor which is to be evaluated in model context
                pred = self.Delta_F[n]*self.magnification(t_grids[n]) +\
                        self.F_base[n]
                
                predictions[n][i] = xo.eval_in_model(pred, sample)

        return predictions 

    def evaluate_prior_model_on_grid(self, trace, t_grids):
        # List of numpy arrays with shape(n_samples, n_datapoints) array with 
        # model predictions
        n_samples = len(np.atleast_1d(trace['t0']))

        predictions = [np.zeros((n_samples, 
            int(T.shape(t_grid).eval()))) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate model for each sample
            for i in range(n_samples):
                # Tensor which is to be evaluated in model context
                pred_mean = self.Delta_F[n]*self.magnification(t_grids[n]) +\
                        self.F_base[n]

                if (n_samples==1):
                    sample = {key:trace[key] for key in trace.keys()}
                else:
                    sample = {key:trace[key][i] for key in trace.keys()}

                predictions[n][i] = xo.eval_in_model(pred_mean, sample)

        return predictions 

    def evaluate_map_model_on_grid(self, t_grids, map_point):
        # List of numpy arrays with shape(n_samples, n_datapoints) array with 
        # model predictions
        predictions = [np.zeros(T.shape(t_grid).eval()) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Tensor which is to be evaluated in model context
            pred = self.Delta_F[n]*self.magnification(t_grids[n]) +\
                    self.F_base[n]

            predictions[n] = xo.eval_in_model(pred, map_point)
            
        return predictions 

class PointSourcePointLensAnnualParallax(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant'):
        super(PointSourcePointLensAnnualParallax, self).__init__(data)

        # Load orbital data from JPL Horizons
        start = Time(data.tables[0]['HJD'][0], format='jd')
        stop = Time(data.tables[0]['HJD'][-1], format='jd')

        epochs={'start':start.iso[:10], 'stop':stop.iso[:10], 'step':'6h'}
        obj = Horizons(id='399', id_type='id', epochs=epochs)

        elements = obj.elements()

        # Calculate the projection of Eath-Sun position vector on the plane
        # of the sky

        # Transform equatorial event coordinates into ecliptic coordinates
        event_ecl_coord = self.equatorial_to_ecliptic_coordinates(
            data.coordinates, 
            0.5*(data.tables[0]['HJD'][0] + data.tables[0]['HJD'][-1]))
        lambda_0  = event_ecl_coord['l']
        beta_0  = event_ecl_coord['b']

        t = np.array(elements['datetime_jd']) # JD
        e = np.array(elements['e'])
        tp = np.array(elements['Tp_jd']) # JD
        n = (2*np.pi/365.25) # mean motion
        Phi_gamma = (77.86)*np.pi/180 # true anomaly at vernal eq. on J2000

        r_sun = 1 - e*np.cos(n*(t - tp)) # to 1st order in e
        lambda_sun = n*(t - tp) - Phi_gamma + 2*e*np.sin(n*(t - tp)) # to 1st order in e

        gamma_w = r_sun*np.sin((lambda_sun - lambda_0))
        gamma_n = r_sun*np.sin(beta_0)*np.cos(lambda_sun - lambda_0)

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


        # Need to define an interpolator for all gamma functions for
        points = [t] # Times for which we have JPL Horizons data
        gamma_w_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_w[:, None])
        gamma_n_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_n[:, None])
        gamma_w_dot_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_w_dot[:, None])
        gamma_n_dot_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_n_dot[:, None])
        gamma_w_ddot_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_w_dot[:, None])
        gamma_n_ddot_interp = xo.interp.RegularGridInterpolator(points, 
            gamma_n_dot[:, None])

        # Interpolate gamma functions onto the grid of observed times, this 
        # happens only once when the model is initialized
        n_max =  np.max([len(table) for table in data.tables])
        self.gamma_w = T._shared(np.stack([np.pad(
            gamma_w_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))
        self.gamma_n = T._shared(np.stack([np.pad(
            gamma_n_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))

        self.gamma_w_dot = T._shared(np.stack([np.pad(
            gamma_w_dot_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))
        self.gamma_n_dot = T._shared(np.stack([np.pad(
            gamma_n_dot_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))

        self.gamma_w_ddot = T._shared(np.stack([np.pad(
            gamma_w_ddot_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))
        self.gamma_n_ddot = T._shared(np.stack([np.pad(
            gamma_n_ddot_interp.evaluate(np.array(table['HJD'])[:, None]).eval().ravel(),
            (0, n_max - len(table['HJD'])), 'constant', 
            constant_values=(0.,)) for table in data.tables]))

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize linear parameters
        self.Delta_F = BoundedNormal('Delta_F', 
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
        self.t0 = pm.Uniform('t0', T.min(self.t[0][self.mask[0].nonzero()]), 
            T.max(self.t[0][self.mask[0].nonzero()]), 
            testval=self.t0_guess(data))
        self.u0 = pm.Normal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        ## Save log prior for each parameter for hierarhical modeling 
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            BoundedNormal.dist(
                mu=T.zeros((self.n_bands, 1)),
                sd=50.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(
                T.min(self.t[0][self.mask[0].nonzero()]), 
                T.max(self.t[0][self.mask[0].nonzero()])).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            BoundedNormal.dist(mu=0., sd=1.5).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))

        # Parallax parameters
        self.gamma_w_t0 = gamma_w_interp.evaluate(\
            T.reshape(self.t0 + 2450000, (1, 1)))[0, 0] 
        self.gamma_n_t0 = gamma_n_interp.evaluate(\
            T.reshape(self.t0 + 2450000, (1, 1)))[0, 0]
        self.gamma_w_dot_t0 = gamma_w_dot_interp.evaluate(\
            T.reshape(self.t0 + 2450000, (1, 1)))[0, 0]
        self.gamma_n_dot_t0 = gamma_n_dot_interp.evaluate(\
            T.reshape(self.t0 + 2450000, (1, 1)))[0, 0]
        self.gamma_w_ddot_t0 = gamma_w_ddot_interp.evaluate(\
            T.reshape(self.t0 + 2450000, (1, 1)))[0, 0]
        self.gamma_n_ddot_t0 = gamma_n_ddot_interp.evaluate(\
            T.reshape(self.t0 + 2450000, (1, 1)))[0, 0]

        self.a_par = pm.Normal('a_par', mu=0, sd=5, testval=0.1)
        self.a_vert = pm.Normal('a_vert', mu=0, sd=5, testval=0.1)

        # Compute the likelihood function
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base # mean function

        # Residuals
        self.r = self.F - mean_func

        if (errorbar_rescaling=='constant'):
            # Define custom prior distributions 
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
            BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

            ## Noise model parameters
            self.A = BoundedNormal1('A', 
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
            self.varF = T.pow(self.A*self.sigF, 2) 

        if (errorbar_rescaling=='additive_variance'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
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
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(self.B, 2)

        if (errorbar_rescaling=='flux_dependant'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
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
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(mag*self.B, 2)

        # Compute the log-likelihood which is additive across different bands
        ll = 0 
        for i in range(self.n_bands):
            ll += self.log_likelihood_single_band(self.r[i], 
                self.varF[i], mag[i], self.mask[i])

        pm.Potential('log_likelihood', ll)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

#    def calculate_gamma_vector_and_derivatives(self, t, tp, e, n, lambda_0, beta_0):
#        Phi_gamma = (77.86)*np.pi/180 # true anomaly at vernal eq. on J2000
#
#        r_sun = 1 - e*np.cos(n*(t - tp)) # to 1st order in e
#        lambda_sun = n*(t - tp) - Phi_gamma + 2*e*np.sin(n*(t - tp)) # to 1st order in e
#
#        gamma_w = r_sun*np.sin((lambda_sun - lambda_0))
#        gamma_n = r_sun*np.sin(beta_0)*np.cos(lambda_sun - lambda_0)
#
#        gamma_w_dot = n*((1 + e*np.cos(n*(t - tp)))*\
#            np.cos(lambda_sun - lambda_0)+\
#            e*np.sin(n*(t - tp))*np.sin(lambda_sun - lambda_0))
#        gamma_n_dot = -n*np.sin(beta_0)*((1 + e*np.cos(n*(t - tp)))\
#            *np.sin(lambda_sun - lambda_0) -\
#            e*np.sin(n*(t - tp))*np.cos(lambda_sun - lambda_0))
#
#        gamma_w_ddot = -n**2*(1 + 2*e*np.cos(n*(t - tp)))*\
#            np.sin(lambda_sun - lambda_0)
#        gamma_n_ddot = -n**2*np.sin(beta_0)*(1 + 2*e*np.cos(n*(t - tp)))*\
#            np.cos(lambda_sun - lambda_0)
#
#        return gamma_w, gamma_n, gamma_w_dot, gamma_n_dot, gamma_w_ddot, gamma_n_ddot

    def calculate_u(self, t):
        T.printing.Print('u0')(self.u0)
        sign = ifelse(T.lt(self.u0, 0.), -1., 1.)
        T.printing.Print('sign')(sign)

        p_E = sign*T.sqrt((self.a_par**2 +\
             self.a_vert**2)/(self.gamma_w_ddot**2 + self.gamma_n_ddot**2))

        sin_phi = p_E*(self.a_par*self.gamma_n_ddot_t0 -\
             self.a_vert*self.gamma_w_ddot_t0)/(self.a_par**2 + self.a_vert**2)

        cos_phi = p_E*(self.a_par*self.gamma_w_ddot_t0 +\
             self.a_vert*self.gamma_n_ddot_t0)/(self.a_par**2 + self.a_vert**2)

        omega_E = 1/self.tE

        u_w = self.u0*(-sin_phi + p_E*(self.gamma_w - self.gamma_w_t0)) +\
            (omega_E*cos_phi - p_E*self.gamma_w_dot_t0)*(t - self.t0)
        u_n = self.u0*(cos_phi + p_E*(self.gamma_w - self.gamma_w_t0)) +\
            (omega_E*sin_phi - p_E*self.gamma_n_dot_t0)*(t - self.t0)

        return T.sqrt(u_w**2 + u_n**2)

    def magnification(self, t):
        u = self.calculate_u(t)
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

    def log_likelihood_single_band(self, r, varF, mag, mask):
        """
        Implements a white noise Gaussian likelihood function, assuming
        that the observations in each photometric band are independent. 
        """
        # Gaussian likelihood
        ll = -0.5*T.sum(T.pow(r[mask.nonzero()], 2.)/varF[mask.nonzero()]) -\
                0.5*T.sum(T.log(2*np.pi*varF[mask.nonzero()]))

        return ll
        
    def evaluate_posterior_model_on_grid(self, trace, t_grids, n_samples=50):
        # List of numpy arrays with shape(n_samples, n_datapoints) array with 
        # model predictions
        predictions = [np.zeros((n_samples, 
            int(T.shape(t_grid).eval()))) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate model for each sample
            for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                    size=n_samples)):
                # Tensor which is to be evaluated in model context
                pred = self.Delta_F[n]*self.magnification(t_grids[n]) +\
                        self.F_base[n]
                
                predictions[n][i] = xo.eval_in_model(pred, sample)

        return predictions 

    def evaluate_prior_model_on_grid(self, trace, t_grids):
        # List of numpy arrays with shape(n_samples, n_datapoints) array with 
        # model predictions
        n_samples = len(np.atleast_1d(trace['t0']))

        predictions = [np.zeros((n_samples, 
            int(T.shape(t_grid).eval()))) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate model for each sample
            for i in range(n_samples):
                # Tensor which is to be evaluated in model context
                pred_mean = self.Delta_F[n]*self.magnification(t_grids[n]) +\
                        self.F_base[n]

                if (n_samples==1):
                    sample = {key:trace[key] for key in trace.keys()}
                else:
                    sample = {key:trace[key][i] for key in trace.keys()}

                predictions[n][i] = xo.eval_in_model(pred_mean, sample)

        return predictions 

    def evaluate_map_model_on_grid(self, t_grids, map_point):
        # List of numpy arrays with shape(n_samples, n_datapoints) array with 
        # model predictions
        predictions = [np.zeros(T.shape(t_grid).eval()) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Tensor which is to be evaluated in model context
            pred = self.Delta_F[n]*self.magnification(t_grids[n]) +\
                    self.F_base[n]

            predictions[n] = xo.eval_in_model(pred, map_point)
            
        return predictions 
    
    def equatorial_to_ecliptic_coordinates(self, coordinates, obs_time):
        # Earth's obliquity angle
        T = (obs_time - 2451545)/365.25/100 # julian centuries since J2000
        eps = 23.439279 - (46.815/60**2)*T
        R = np.array([[1, 0, 0],
                    [0, np.cos(eps), np.sin(eps)],
                    [0, -np.sin(eps), np.cos(eps)]])
        
        ra = coordinates.ra.to(u.rad).value
        dec = coordinates.dec.to(u.rad).value
        coord_ec_cartesian = np.array([np.cos(ra)*np.cos(dec),
                                    np.sin(ra)*np.cos(dec),
                                    np.sin(dec)])
        
        # Transformation between equatorial and ecliptical coordinates
        coord_eq_cartesian = R @ coord_ec_cartesian[:, np.newaxis]
        
        b = np.arcsin(coord_eq_cartesian[2])
        l = 2*np.arctan(coord_eq_cartesian[1]/(np.cos(b) + coord_eq_cartesian[0]))
        
        return {'l':l, 'b':b}

class PointSourcePointLensMatern32(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant'):
        super(PointSourcePointLensMatern32, self).__init__(data)

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize linear parameters
        self.Delta_F = BoundedNormal('Delta_F', 
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
        self.t0 = pm.Uniform('t0', T.min(self.t[0][self.mask[0].nonzero()]), 
            T.max(self.t[0][self.mask[0].nonzero()]), 
            testval=self.t0_guess(data))
        self.u0 = BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        ## Save log prior for each parameter for hierarhical modeling 
        self.logp_Delta_F = pm.Deterministic('logp_Delta_F',
            BoundedNormal.dist(
                mu=T.zeros((self.n_bands, 1)),
                sd=50.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(
                T.min(self.t[0][self.mask[0].nonzero()]), 
                T.max(self.t[0][self.mask[0].nonzero()])).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            BoundedNormal.dist(mu=0., sd=1.5).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))

        # Compute the likelihood function
        mag = self.magnification(self.t) 
        mean_func = self.Delta_F*mag +  self.F_base # mean function

        # Residuals
        self.r = self.F - mean_func

        if (errorbar_rescaling=='constant'):
            # Define custom prior distributions 
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
            BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

            ## Noise model parameters
            self.A = BoundedNormal1('A', 
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
            self.varF = T.pow(self.A*self.sigF, 2) 

        if (errorbar_rescaling=='additive_variance'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=1*T.ones((self.n_bands, 1)),
                testval=0.01*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    testval=1.5*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            self.logp_B = pm.Deterministic('logp_B',
                pm.Normal.dist(
                    mu=T.zeros((self.n_bands, 1)), 
                    sd=1*T.ones((self.n_bands, 1)),
                    testval=0.01*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.B))

            # Diagonal terms of the covariance matrix
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(self.B, 2)

        if (errorbar_rescaling=='flux_dependant'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=5*T.ones((self.n_bands, 1)),
                testval=0.01*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    testval=1.5*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            self.logp_B = pm.Deterministic('logp_B',
                pm.Normal.dist(
                    mu=T.zeros((self.n_bands, 1)), 
                    sd=5*T.ones((self.n_bands, 1)),
                    testval=0.01*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.B))

            # Diagonal terms of the covariance matrix
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(mag*self.B, 2)

        ## Compute parameters for the prior on GP hyperparameters
        tables = data.get_standardized_data()
        invgamma_a = np.zeros(self.n_bands)
        invgamma_b = np.zeros(self.n_bands)

        for i in range(len(tables)):
            t = np.array(tables[i]['HJD'])
            invgamma_a[i], invgamma_b[i] = fsolve(self.solve_for_invgamma_params, 
                (0.1, 0.1), (np.median(np.diff(t)), t[-1] - t[0]))

        self.sigma = BoundedNormal('sigma', 
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
            BoundedNormal.dist(
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
        
        # Compute the log-likelihood which is additive across different bands
        ll = 0 
        for i in range(self.n_bands):
            ll += self.log_likelihood_single_band(self.t[i], self.r[i], 
                self.varF[i], self.sigma[i], self.rho[i], self.mask[i])

        pm.Potential('log_likelihood', ll)

        # Define helpful class attributes
        self.free_parameters = [RV.name for RV in self.basic_RVs]
        self.initial_logps = [RV.logp(self.test_point) for RV in self.basic_RVs]

    def magnification(self, t):
        u = T.sqrt(self.u0**2 + ((t - self.t0)/self.tE)**2)
        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))

        return (A(u) - 1)/(A(self.u0) - 1) 
        
    def solve_for_invgamma_params(self, params, x_min, x_max):
        """
        Returns parameters of an inverse gamma distribution p(x) such that 
        0.1% of total prob. mass is assigned to values of x < x_min and 
        1% of total prob. masss  to values greater than x_max.
        """

        def inverse_gamma_cdf(x, alpha, beta):
            return invgamma.cdf(x, alpha, scale=beta)

        lower_mass = 0.01
        upper_mass = 0.99

        # Trial parameters
        alpha, beta = params

        # Equation for the roots defining params which satisfy the constraint
        return (inverse_gamma_cdf(x_min, alpha, beta) - \
            lower_mass, inverse_gamma_cdf(x_max, alpha, beta) - upper_mass)

    def log_likelihood_single_band(self, t, r, varF, sigma, rho, mask):
        # Calculate likelihood
        kernel = terms.Matern32Term(sigma=sigma, rho=rho)
        # The exoplanet.gp.GP constructor takes an optional argument J which 
        # specifies the width of the problem if it is known at compile time. 
        # This is actually two times the J from the celerite paper
        gp = GP(kernel, t[mask.nonzero()], varF[mask.nonzero()], J=2) # J=2 for Matern32 kernel

        # Add a custom "potential" (log probability function) with the 
        # GP likelihood
        return gp.log_likelihood(r[mask.nonzero()])
        
    def evaluate_posterior_model_on_grid(self, trace, t_grids, n_samples=50):
        predictions = [np.zeros((n_samples, 
            int(T.shape(t_grid).eval()))) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate model for each sample in the chain
            for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                    size=n_samples)):
                # Evaluate mean function at observed times
                mag_obs = self.magnification(self.t[n][self.mask[n].nonzero()])
                mean_func_obs = self.Delta_F[n]*mag_obs + self.F_base[n]

                # Evaluate mean function on fine grid
                mag = self.magnification(t_grids[n]) 
                mean_func = self.Delta_F[n]*mag + self.F_base[n]
                
                # Diagonal terms of the covariance matrix
                varF = T.pow(self.A[n]*self.sigF[n][self.mask[n].nonzero()], 2) 
#                     T.pow(mag_obs*self.B[n], 2)

                # Initialize kernel and gp object
                kernel = terms.Matern32Term(sigma=self.sigma[n], 
                    rho=self.rho[n])
                gp = GP(kernel, self.t[n][self.mask[n].nonzero()], varF, J=2)

                # Calculate log_likelihood 
                r = self.F[n][self.mask[n].nonzero()] - mean_func_obs
                gp.log_likelihood(r)

                # Evaluate tensors in model context
                predictions[n][i] =\
                    xo.eval_in_model(gp.predict(t_grids[n]), sample) +\
                    xo.eval_in_model(mean_func, sample) 
            
        return predictions 

    def evaluate_map_model_on_grid(self, t_grids, map_point):
        predictions = [np.zeros(int(T.shape(t_grid).eval())) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate mean function at observed times
            mag_obs = self.magnification(self.t[n][self.mask[n].nonzero()])
            mean_func_obs = self.Delta_F[n]*mag_obs + self.F_base[n]

            # Evaluate mean function on fine grid
            mag = self.magnification(t_grids[n]) 
            mean_func = self.Delta_F[n]*mag + self.F_base[n]
            
            # Diagonal terms of the covariance matrix
            varF = T.pow(self.A[n]*self.sigF[n][self.mask[n].nonzero()], 2) +\
                    T.pow(mag_obs*self.B[n], 2)

            # Initialize kernel and gp object

            kernel = terms.Matern32Term(sigma=self.sigma[n], 
                rho=self.rho[n])
            gp = GP(kernel, self.t[n][self.mask[n].nonzero()], varF, J=2)

            # Calculate log_likelihood 
            r = self.F[n][self.mask[n].nonzero()] - mean_func_obs
            gp.log_likelihood(r)

            # Evaluate tensors in model context
            predictions[n] =\
                xo.eval_in_model(gp.predict(t_grids[n]), map_point) +\
                xo.eval_in_model(mean_func, map_point) 
        
        return predictions 

class PointSourcePointLensMarginalized(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant'):
        super(PointSourcePointLensMarginalized, self).__init__(data)

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize non-linear parameters
        ## Posterior is multi-modal in t0 and it's critical that the it is 
        ## initialized near the true value
        self.t0 = pm.Uniform('t0', T.min(self.t[0][self.mask[0].nonzero()]), 
            T.max(self.t[0][self.mask[0].nonzero()]), 
            testval=self.t0_guess(data))
        self.u0 = BoundedNormal('u0', mu=0., sd=1.5, testval=0.1)
        self.teff = BoundedNormal('teff', mu=0., sd=365., testval=20.)
        
        # Deterministic transformations
        self.tE = pm.Deterministic("tE", self.teff/self.u0) 

        ## Save log prior for each parameter for hierarhical modeling 
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(
                T.min(self.t[0][self.mask[0].nonzero()]), 
                T.max(self.t[0][self.mask[0].nonzero()])).logp(self.t0))
        self.logp_u0 = pm.Deterministic('logp_u0', 
            BoundedNormal.dist(mu=0., sd=1.5).logp(self.u0))
        self.logp_teff = pm.Deterministic('logp_teff', 
            BoundedNormal.dist(mu=0., sd=365.).logp(self.teff))

        # Compute the likelihood function
        self.mag = self.magnification(self.t) 

        if (errorbar_rescaling=='constant'):
            # Define custom prior distributions 
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
            BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            
            # Diagonal terms of the covariance matrix
            self.varF = T.pow(self.A*self.sigF, 2) 

        if (errorbar_rescaling=='additive_variance'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=1*T.ones((self.n_bands, 1)),
                testval=0.01*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    testval=1.5*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            self.logp_B = pm.Deterministic('logp_B',
                pm.Normal.dist(
                    mu=T.zeros((self.n_bands, 1)), 
                    sd=1*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.B))

            # Diagonal terms of the covariance matrix
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(self.B, 2)

        if (errorbar_rescaling=='flux_dependant'):
            ## Noise model parameters
            self.A = BoundedNormal1('A', 
                mu=T.ones((self.n_bands, 1)),
                sd=2.*T.ones((self.n_bands, 1)),
                testval=1.5*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            self.B = BoundedNormal('B', 
                mu=T.zeros((self.n_bands, 1)), 
                sd=5*T.ones((self.n_bands, 1)),
                testval=0.01*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1))

            ## Save log prior for each parameter for hierarhical modeling 
            self.logp_A = pm.Deterministic('logp_A',
                pm.Normal.dist(
                    mu=T.ones((self.n_bands, 1)),
                    sd=2.*T.ones((self.n_bands, 1)),
                    testval=1.5*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.A))
            self.logp_B = pm.Deterministic('logp_B',
                pm.Normal.dist(
                    mu=T.zeros((self.n_bands, 1)), 
                    sd=5*T.ones((self.n_bands, 1)),
                    testval=0.01*T.ones((self.n_bands, 1)),
                    shape=(self.n_bands, 1)).logp(self.B))

            # Diagonal terms of the covariance matrix
            self.varF = T.pow(self.A*self.sigF, 2) + T.pow(mag*self.B, 2)

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