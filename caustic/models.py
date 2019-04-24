import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T

import exoplanet as xo
from exoplanet.gp import terms, GP

from scipy.special import gamma
from scipy.stats import invgamma
from scipy.optimize import fsolve

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

    def evaluate_posterior_model_on_grid(self, trace, t_grid, n_samples=50):
        """
        Evaluates model on dense grid for N_pred random samples from the 
        posterior.
        
        """
        pass

    def evaluate_map_model_on_grid(self, t_grid, map_point):
        """
        Evaluates model on dense grid for N_pred random samples from the 
        posterior.
        
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

        return np.median(times[fluxes > 4])
    
    def generate_mock_data_from_prior(self, event):
        """
        Generates mock light curve by sampling the prior distribution. This
        method should be overloaded by subclasses.
        """

class OutlierRemovalModel(SingleLensModel):
    #  override __init__ function from pymc3 Model class
    def __init__(self, data, name='', model=None):
        super(OutlierRemovalModel, self).__init__()

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize linear parameters
        self.Delta_F = pm.Lognormal('Delta_F', 
            mu=10.*T.ones((self.n_bands, 1)),
            sd=2.*T.ones((self.n_bands, 1)),
            testval=10.*T.ones((self.n_bands, 1)),
            shape=(self.n_bands, 1))

        self.F_base = pm.Normal('F_base', 
            mu=T.zeros((self.n_bands, 1)), 
            sd=0.05*T.ones((self.n_bands, 1)),
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
        self.varF = T.pow(self.A*self.sigF, 2) + T.pow(mag*self.B, 2)

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
   
    def evaluate_map_model_on_grid(self, t_grid, map_point):
        """
        Evaluates GP model on dense grid for N_pred random samples from the 
        posterior.
        
        """
        model_prediction = np.zeros((self.n_bands, len(t_grid)))
        t_grid = T._shared(t_grid)

        for n in range(self.n_bands):
            # Evaluate mean function at observed times
            mag_obs = self.magnification(self.t[n])
            mean_func_obs = self.Delta_F[n]*mag_obs + self.F_base[n]

            # Evaluate mean function on fine grid
            mag = self.magnification(t_grid) 
            mean_func = self.Delta_F[n]*mag + self.F_base[n]
            
            # Diagonal terms of the covariance matrix
            varF = T.pow(self.A[n]*self.sigF[n], 2) +\
                    T.pow(mag_obs*self.B[n], 2)

            # Initialize kernel and gp object

            kernel = terms.Matern32Term(sigma=self.sigma[n], 
                rho=self.rho[n])
            gp = GP(kernel, self.t[n], varF, J=2)

            # Calculate log_likelihood 
            r = self.F[n] - mean_func_obs
            gp.log_likelihood(r)

            # Evaluate tensors in model context
            model_prediction[n] =\
                xo.eval_in_model(gp.predict(t_grid), map_point) +\
                xo.eval_in_model(mean_func, map_point) 
        
        return model_prediction

    def evaluate_posterior_model_on_grid(self, trace, t_grid, n_samples=50):
        """
        Evaluates GP model on dense grid for N_pred random samples from the 
        posterior.
        
        """
        model_prediction = np.zeros((self.n_bands, n_samples, len(t_grid)))
        t_grid = T._shared(t_grid)

        for n in range(self.n_bands):
            # Evaluate model for each sample in the chain
            for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                    size=n_samples)):
                # Evaluate mean function at observed times
                mag_obs = self.magnification(self.t[n])
                mean_func_obs = self.Delta_F[n]*mag_obs + self.F_base[n]

                # Evaluate mean function on fine grid
                mag = self.magnification(t_grid) 
                mean_func = self.Delta_F[n]*mag + self.F_base[n]
                
                # Diagonal terms of the covariance matrix
                varF = T.pow(self.A[n]*self.sigF[n], 2) +\
                     T.pow(mag_obs*self.B[n], 2)

                # Initialize kernel and gp object
                kernel = terms.Matern32Term(sigma=self.sigma[n], 
                    rho=self.rho[n])
                gp = GP(kernel, self.t[n], varF, J=2)

                # Calculate log_likelihood 
                r = self.F[n] - mean_func_obs
                gp.log_likelihood(r)

                # Evaluate tensors in model context
                model_prediction[n, i] =\
                    xo.eval_in_model(gp.predict(t_grid), sample) +\
                    xo.eval_in_model(mean_func, sample) 
            
        return model_prediction

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
            pm.Lognormal.dist(
                mu=10*T.ones((self.n_bands, 1)),
                sd=15.*T.ones((self.n_bands, 1)),
                testval=3.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                testval=T.zeros((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(T.min(self.t[0]), T.max(self.t[0])).logp(self.t0))
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
        # List of tensors with shape (n_datapoints) which are the 
        # times at which the model is to be evaluated in each band
        t_grids_tensors = [t_grid for t_grid in t_grids]

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

    def evaluate_prior_model_on_grid(self, trace, t_grids, n_samples):
        # List of tensors with shape (n_datapoints) which are the 
        # times at which the model is to be evaluated in each band
        t_grids_tensors = [t_grid for t_grid in t_grids]

        # List of numpy arrays with shape(n_samples, n_datapoints) array with 
        # model predictions
        predictions = [np.zeros((n_samples, 
            int(T.shape(t_grid).eval()))) for t_grid in t_grids]

        for n in range(self.n_bands):
            # Evaluate model for each sample
            for i in range(n_samples):
                # Tensor which is to be evaluated in model context
                pred = self.Delta_F[n]*self.magnification(t_grids[n]) +\
                        self.F_base[n]
                sample = {key:trace[key][i] for key in trace.keys()}

                predictions[n][i] = xo.eval_in_model(pred, sample)

        return predictions 

    def evaluate_map_model_on_grid(self, t_grid, map_point):
        model_prediction = np.zeros((self.n_bands, len(t_grid)))
        t_grid = T._shared(t_grid)

        for n in range(self.n_bands):
            # Tensor which is to be evaluated in model context
            pred = self.Delta_F[n]*self.magnification(t_grid) +\
                    self.F_base[n]

            model_prediction[n] = xo.eval_in_model(pred, map_point)
            
        return model_prediction

    


class PointSourcePointLensMatern32(SingleLensModel):
    def __init__(self, data, errorbar_rescaling='constant'):
        super(PointSourcePointLensMatern32, self).__init__(data)

        # Define custom prior distributions 
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0) 
        BoundedNormal1 = pm.Bound(pm.Normal, lower=1.) 

        # Initialize linear parameters
        self.Delta_F = pm.Lognormal('Delta_F', 
            mu=10.*T.ones((self.n_bands, 1)),
            sd=15.*T.ones((self.n_bands, 1)),
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
            pm.Lognormal.dist(
                mu=10*T.ones((self.n_bands, 1)),
                sd=15.*T.ones((self.n_bands, 1)),
                testval=3.*T.ones((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.Delta_F))
        self.logp_F_base = pm.Deterministic('logp_F_base',
            pm.Normal.dist(
                mu=T.zeros((self.n_bands, 1)), 
                sd=0.6*T.ones((self.n_bands, 1)),
                testval=T.zeros((self.n_bands, 1)),
                shape=(self.n_bands, 1)).logp(self.F_base))
        self.logp_t0 = pm.Deterministic('logp_t0',
            pm.Uniform.dist(T.min(self.t[0]), T.max(self.t[0])).logp(self.t0))
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
        
    def evaluate_posterior_model_on_grid(self, trace, t_grid, n_samples=50):
        """
        Evaluates GP model on dense grid for N_pred random samples from the 
        posterior.
        
        """
        model_prediction = np.zeros((self.n_bands, n_samples, len(t_grid)))
        t_grid = T._shared(t_grid)

        for n in range(self.n_bands):
            # Evaluate model for each sample in the chain
            for i, sample in enumerate(xo.get_samples_from_trace(trace, 
                    size=n_samples)):
                # Evaluate mean function at observed times
                mag_obs = self.magnification(self.t[n])
                mean_func_obs = self.Delta_F[n]*mag_obs + self.F_base[n]

                # Evaluate mean function on fine grid
                mag = self.magnification(t_grid) 
                mean_func = self.Delta_F[n]*mag + self.F_base[n]
                
                # Diagonal terms of the covariance matrix
                varF = T.pow(self.A[n]*self.sigF[n], 2) +\
                     T.pow(mag_obs*self.B[n], 2)

                # Initialize kernel and gp object
                kernel = terms.Matern32Term(sigma=self.sigma[n], 
                    rho=self.rho[n])
                gp = GP(kernel, self.t[n], varF, J=2)

                # Calculate log_likelihood 
                r = self.F[n] - mean_func_obs
                gp.log_likelihood(r)

                # Evaluate tensors in model context
                model_prediction[n, i] =\
                    xo.eval_in_model(gp.predict(t_grid), sample) +\
                    xo.eval_in_model(mean_func, sample) 
            
        return model_prediction

    def evaluate_map_model_on_grid(self, t_grid, map_point):
        """
        Evaluates GP model on dense grid for N_pred random samples from the 
        posterior.
        
        """
        model_prediction = np.zeros((self.n_bands, len(t_grid)))
        t_grid = T._shared(t_grid)

        for n in range(self.n_bands):
            # Evaluate mean function at observed times
            mag_obs = self.magnification(self.t[n])
            mean_func_obs = self.Delta_F[n]*mag_obs + self.F_base[n]

            # Evaluate mean function on fine grid
            mag = self.magnification(t_grid) 
            mean_func = self.Delta_F[n]*mag + self.F_base[n]
            
            # Diagonal terms of the covariance matrix
            varF = T.pow(self.A[n]*self.sigF[n], 2) +\
                    T.pow(mag_obs*self.B[n], 2)

            # Initialize kernel and gp object

            kernel = terms.Matern32Term(sigma=self.sigma[n], 
                rho=self.rho[n])
            gp = GP(kernel, self.t[n], varF, J=2)

            # Calculate log_likelihood 
            r = self.F[n] - mean_func_obs
            gp.log_likelihood(r)

            # Evaluate tensors in model context
            model_prediction[n] =\
                xo.eval_in_model(gp.predict(t_grid), map_point) +\
                xo.eval_in_model(mean_func, map_point) 
        
        return model_prediction

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
            pm.Uniform.dist(T.min(self.t[0]), T.max(self.t[0])).logp(self.t0))
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