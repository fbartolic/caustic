import numpy as np
import pymc3 as pm
import theano.tensor as T

import exoplanet as xo

from .utils import construct_masked_tensor

from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy import units as u

class Trajectory:
    """
    Computes the separation between the lens and the source star as a 
    function of time assuming that either no parllax effects or annual parallax
    parametrized with pi_EE and pi_EN parameters.
    """
    def __init__(
        self,
        data=None,
        t_0=None,
        u_0=None,
        t_E=None,
        pi_EE=None,
        pi_EN=None
    ):
        self.data = data
        self.t_0 = t_0
        self.u_0 = u_0
        self.t_E = t_E
        self.pi_EE = pi_EE
        self.pi_EN = pi_EN

        if (self.pi_EE or self.pi_EN) is not None:
            # Call NASA's JPL Horizons and compute orbital Earth's orbit at observed 
            # times
            t_min = np.min([table['HJD'][0] for table in data.light_curves])
            t_max = np.max([table['HJD'][-1] for table in data.light_curves])

            t = np.linspace(t_min, t_max, 5000) 
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
            zeta_e, zeta_n = self.__project_vector_on_sky(s_t, data.event_coordinates)
            zeta_e_dot, zeta_n_dot = self.__project_vector_on_sky(v_t, 
                data.event_coordinates)
            zeta_e_ddot, zeta_n_ddot = self.__project_vector_on_sky(a_t, 
                data.event_coordinates)

            # Initialize `exoplanet.interp.RegularGridInterpolator` objects 
            # These interpolators are used to evaluate the zeta function and its
            # derivatives at arbitrary times which is needed to compute the 
            # likelihood.
            points = [t - 2450000]  # Times used for interpolation

            # zeta(t) interpolators
            self.zeta_e_interp = xo.interp.RegularGridInterpolator(points, 
                zeta_e[:, None])
            self.zeta_n_interp= xo.interp.RegularGridInterpolator(points, 
                zeta_n[:, None])

            # dot[zeta(t)] interpolators
            self.zeta_e_dot_interp = xo.interp.RegularGridInterpolator(points, 
                zeta_e_dot[:, None])
            self.zeta_n_dot_interp = xo.interp.RegularGridInterpolator(points, 
                zeta_n_dot[:, None])

            # ddot[zeta(t)] interpolators
            self.zeta_e_ddot_interp = xo.interp.RegularGridInterpolator(points, 
                zeta_e_ddot[:, None])
            self.zeta_n_ddot_interp = xo.interp.RegularGridInterpolator(points, 
                zeta_n_ddot[:, None])

    def __project_vector_on_sky(self, matrix, coordinates):
        """
        This function takes a 3D cartesian vector specified
        in the ICRS coordinate system and evaluated at differrent
        times (t_i,...t_N) and projects it onto a spherical coordinate
        system on the plane of the sky with the origin at the position
        defined by the coordinates.
        
        Parameters
        ----------
        matrix : ndarray
            Matrix of shape (len(times), 3)
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

    def __compute_delta_zeta(self, t):
        """
        Computes the components of delta_zeta vector - the devaition of the 
        projected separation of the Sun relative to time t_0.

        Parameters
        ----------
        t : theano.tensor
            Tensor containing times for which u(t) is to be computed. Needs 
            to have shape (n_bands, npoints)
       
        Returns
        -------
        list  
            List of theano.tensor containing North and East components of 
            delta_zeta(t). Tensors are of the same shape as t.
        """
        zeta_e_list = []
        zeta_n_list = []

        for idx in range(len(self.data.light_curves)):
            # the interpolation function requires tensor of shape (n_points, 1)
            # as an input 
            pts = T.reshape(t[idx], (T.shape(t[idx])[0], 1))
            zeta_e_list.append(self.zeta_e_interp.evaluate(pts).transpose())
            zeta_n_list.append(self.zeta_n_interp.evaluate(pts).transpose())

        # Stack interpolated functions such that they have the same shape as
        # self.t, namely, (n_bands, npoints)
        zeta_e = T.concatenate(zeta_e_list, axis=1)
        zeta_n = T.concatenate(zeta_n_list, axis=1)

        zeta_e_t_0 = self.zeta_e_interp.evaluate(T.reshape(self.t_0,
            (1, 1)))[0, 0]
        zeta_n_t_0 = self.zeta_n_interp.evaluate(T.reshape(self.t_0,
            (1, 1)))[0, 0]
        zeta_e_dot_t_0 = self.zeta_e_dot_interp.evaluate(T.reshape(self.t_0,
            (1, 1)))[0, 0]
        zeta_n_dot_t_0 = self.zeta_n_dot_interp.evaluate(T.reshape(self.t_0,
            (1, 1)))[0, 0]

        # Compute delta_zeta function 
        delta_zeta_e = zeta_e - zeta_e_t_0 -\
            (t - self.t_0)*zeta_e_dot_t_0
        delta_zeta_n = zeta_n - zeta_n_t_0 -\
            (t - self.t_0)*zeta_n_dot_t_0

        return delta_zeta_n, delta_zeta_e

    def compute_trajectory(self, t, return_components=False):
        """
        Computes the magnitude of the relative lens-source separation vector
        u(t).         

        Parameters
        ----------
        t : theano.tensor
            Tensor containing times for which u(t) is to be computed. Needs 
            to have shape (n_bands, npoints)
        return_components : bool, optional
            If true, the function returns the vector components (u_n, u_e)
            of u(t) rather than its magnitude, by default False.

        Returns
        -------
        theano.tensor
            The magnitude of the separation vector u(t) for each time or a 
            tuple with the two components of u(t) in (north, east) basis.
        """
        if (self.pi_EE or self.pi_EN) is None:
            u = T.sqrt(self.u_0**2 + ((t - self.t_0)/self.t_E)**2)
            return u

        elif return_components==True:
            delta_zeta_n, delta_zeta_e = self.__compute_delta_zeta(t)

            pi_E = T.sqrt(self.pi_EE**2 + self.pi_EN**2)

            cospsi = self.pi_EN/pi_E
            sinpsi = self.pi_EE/pi_E

            u_n = -self.u_0*sinpsi + (t - self.t_0)/\
                self.t_E*cospsi + pi_E*delta_zeta_n
            u_e = self.u_0*cospsi + (t - self.t_0)/\
                self.t_E*sinpsi + pi_E*delta_zeta_e

            return u_n, u_e

        else:
            delta_zeta_n, delta_zeta_e = self.__compute_delta_zeta(t)

            u_per = self.u_0 + self.pi_EN*delta_zeta_e - self.pi_EE*delta_zeta_n
            u_par = (t - self.t_0)/self.t_E + self.pi_EE*delta_zeta_e +\
                self.pi_EN*delta_zeta_n

            return T.sqrt(u_par**2 + u_per**2)