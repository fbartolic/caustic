import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
import copy


from .utils import construct_masked_tensor, estimate_t0


import exoplanet as xo
from exoplanet.gp import terms, GP


from scipy.special import zeta
from scipy.stats import invgamma
from scipy.optimize import fsolve


class SingleLensModel(pm.Model):
    """
    Abstract class for a single lens model.  Subclasses should implement the
    should implement the :func:`compute_magnification` method. 

    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Caustic data object.
    standardize : bool
        If ``True``, the fluxes for each band will be independently rescaled
        to zero medain and unit standard deviation. This generally improves 
        the sampling efficiency. By default ``True``.
    """
    #  override __init__ function from pymc3 Model class
    def __init__(
        self, 
        data=None, 
        standardize=True
    ):
        super(SingleLensModel, self).__init__()

        if standardize==True:
            # Load data standardized to zero median and unit variance
            tables = data.get_standardized_data()
        else:
            tables = data.get_standardized_data(rescale=False)

        self.standardized_data = standardize
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

    def compute_magnification(self, u, u0):
        """
        Computes the magnification fraction :math:`[A(u) - 1]/[A(u_0) - 1]`
        where :math:`A(u)` is the magnification function.
        
        Parameters
        ----------
        u : theano.tensor
            Trajectory of the lens :math:`u(t)` with respect to the source 
            in units of ang. Einstein radii.
        u0 : theano.tensor
            Lens-source separation at time :math:`t_0`.

        Returns
        -------
        theano.tensor
            The value of the magnification at each time t.
        """
        A_u = (u**2 + 2)/(u*T.sqrt(u**2 + 4))
        A_u0 = (u0**2 + 2)/(T.abs_(u0)*T.sqrt(u0**2 + 4))

        return (A_u - 1)/(A_u0 - 1)

    def compute_log_likelihood(self, r, var_F, gp_list=None, **kwargs):
        """"
        Computes the total log likelihood of the model assuming that the 
        observations in different bands are independent. The likelihood is 
        assumed to be multivariate Gaussian with an optional
        Gaussian Process modeling the covariance matrix elements.

        Parameters
        ----------
        r : theano.tensor
            Residuals of the mean model with respect to the data.
        var_F : theano.tensor
            Diagonal elements of the covariance matrix. 
        gp_list : list
            List of ``exoplanet.gp.GP`` objects, one per each band. If these
            are provided the likelihood which is computed is the GP marginal
            likelihood.

        Returns
        -------
        theano.tensor 
            Log likelihood.
        """
        ll = 0 
        # Iterate over all observed bands
        for i in range(self.n_bands):
            r_ = r[i][self.mask[i].nonzero()]
            var_F_ = var_F[i][self.mask[i].nonzero()]

            if (gp_list==None):
                ll += -0.5*T.sum(r_**2/var_F_) -\
                    0.5*T.sum(T.log(2*np.pi*var_F_))
            else:
                ll += gp_list[i].log_likelihood(r_)
        return ll
    
    def compute_marginalized_log_likelihood(self, magnification, var_F, L):
        """
        Computes the multivariate guassian log-likelihood analytically 
        marginalized over the linear flux parameters. 
        
        Parameters
        ----------
        magnification : theano.tensor
            Computed magnification, shape ``(n_bands, max_npoints)``.
        var_F : theano.tensor
            Diagonal elements of the covariance matrix, shape ``(n_bands, 
            max_npoints)``.
        L : theano.tensor
            Covariance matrix for the multivariate gaussian prior on the linear
            flux parameters. The Gaussian prior is a requirement to make
            the analytical marginalization tractable. The prior is assumed to
            be equal for all bands, hence the matrix needs to be of shape 
            ``(2, 2)``.
        """
        def log_likelihood_single_band(F, var_F, mag, mask):
            F = F[mask.nonzero()]
            var_F = var_F[mask.nonzero()]
            mag = mag[mask.nonzero()]

            N = T.shape(F)[0]

            # Linear parameter matrix
            A = T.stack([mag, T.ones_like(F)], axis=1)

            # Covariance matrix
            C_diag = var_F
            C = T.nlinalg.diag(C_diag)

            # Calculate inverse of covariance matrix for marginalized likelihood
            inv_C = T.nlinalg.diag(T.pow(C_diag, -1.))
            ln_detC = T.log(C_diag).sum()

            inv_L = T.nlinalg.diag(T.pow(T.nlinalg.diag(L), -1.))
            ln_detL = T.log(T.nlinalg.diag(L)).sum()

            S = inv_L + T.dot(A.transpose(), T.dot(inv_C, A))
            inv_S = T.nlinalg.matrix_inverse(S)
            ln_detS = T.log(T.nlinalg.det(S))

            inv_SIGMA =inv_C -\
                T.dot(inv_C, T.dot(A, T.dot(inv_S, T.dot(A.transpose(), inv_C))))
            ln_detSIGMA = ln_detC + ln_detL + ln_detS

            # Calculate marginalized likelihood
            ll = -0.5*T.dot(F.transpose(), T.dot(inv_SIGMA, F)) -\
                0.5*N*np.log(2*np.pi) - 0.5*ln_detSIGMA

            return ll

        # Compute the log-likelihood which is additive across different bands
        ll = 0 
        for i in range(self.n_bands):
            ll += log_likelihood_single_band(self.F[i], 
                var_F[i], magnification[i], self.mask[i])

        return ll 

    def generate_mock_dataset(self):
        """
        Generates mock :func:`~caustic.data.Data` object by sampling the prior
        predictive distribution.
        """
        raise NotImplementedError()