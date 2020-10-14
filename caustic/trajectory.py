import exoplanet as xo
import numpy as np
import theano.tensor as tt
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time


class Trajectory:
    """
    Computes the magnitude of the separation vector :math:`\mathbf{u}(t)` 
    between the lens and the source star as a function of time assuming that 
    either no parallax effects or annual parallax parametrized using one three
    different parametrizations. The standard parametrization in terms of the
    "microlensing parallax vector" components
    :math:`\pi_\mathrm{EE}` and :math:`\pi_\mathrm{EN}`. The parametrization in
    terms of the relative lens-source parallax :math:`\pi_\mathrm{E}` and an
    angle :math:`\psi` or the parametrization in terms of the local acceleration
    of the lens at time t0 :math:`(a_\bot, a_\parallel)`. If you are planning 
    to use the angle parametrization we recommend using the 
    :code:`exoplanet.distributions.Angle` object to avoid discontinuities
    at :math:`\pi`.

    Example usage:

    .. code-block:: python

        trajectory = caustic.trajectory.Trajectory(event, t0, u0, tE)

        u = trajectory.compute_trajectory(model.t)

    including parallax

    .. code-block:: python

        trajectory = caustic.trajectory.Trajectory(event, t0, u0, tE,
            piEE=piEE, piEN=piEN)

        u = trajectory.compute_trajectory(model.t)
    
    or

    .. code-block:: python

        trajectory = caustic.trajectory.Trajectory(event, t0, u0, tE,
            piE=piE, psi=psi)

        u, un, ue = trajectory.compute_trajectory(model.t, return_components=True)

    or

    .. code-block:: python

        trajectory = caustic.trajectory.Trajectory(event, t0, u0, tE,
            a_per=a_per, a_par=a_par)
        
        u, un, ue = trajectory.compute_trajectory(model.t, return_components=True)

    Parameters
    ----------
    data : :func:`~caustic.data.Data`
        Microlensing event data object.
    t0 : :func:`~pymc3.model.FreeRV`
        t0 parameter.
    u0 : :func:`~pymc3.model.FreeRV`
        u0 parameter, needs to be positive if parallax parameters aren't provided.
    tE : :func:`~pymc3.model.FreeRV`
        tE parameter.
    piEE : :func:`~pymc3.model.FreeRV`
        The :math:`\pi_mathrm{EE}` components of the microlensing parallax
        "vector".
    piEN : :func:`~pymc3.model.FreeRV`
        The :math:`\pi_mathrm{EN}` components of the microlensing parallax
        "vector".
    piE : :func:`~pymc3.model.FreeRV`
        The relative lens-source parallax
        :math:`\pi_\mathrm{E}=\sqrt{\pi_\mathrm{EE}^2 +\pi_\mathrm{EN}^2}`.
    psi : :func:`~pymc3.model.FreeRV`
        The angle between the ecliptic coordinate system and the coordinate
        system tangential to the lens trajectory at time :math:`t=t_0`. We have
        :math:`\pi_mathrm{EE}=\pi_\mathrm{E}\sin\psi` and
        :math:`\pi_mathrm{EN}=\pi_\mathrm{E}\cos\psi`.
    a_per : :func:`~pymc3.model.FreeRV`
        The perpendicular component :math:`a_\bot` of the apparent lens
        acceleration at time :math:`t=t_0` in units :math:`\mathrm{yr}^{-2}`.
    a_par : :func:`~pymc3.model.FreeRV`
        The parallel component :math:`a_\parallel` of the apparent lens
        acceleration at time :math:`t=t_0` in units :math:`\mathrm{yr}^{-2}`.
    """

    def __init__(
        self,
        data=None,
        t0=None,
        u0=None,
        tE=None,
        piEE=None,
        piEN=None,
        piE=None,
        psi=None,
        a_par=None,
        a_per=None,
    ):
        self.data = data
        self.t0 = t0
        self.u0 = u0
        self.tE = tE
        self.piEE = piEE
        self.piEN = piEN
        self.piE = piE
        self.psi = psi
        self.a_par = a_par
        self.a_per = a_per

        # Initialize objects needed for computing Earth's trajectory on the plane
        # of the sky if the parallax parameters are provided
        if (
            (self.piEE and self.piEN)
            or (self.piE and self.psi)
            or (self.a_par and self.a_per) is not None
        ):
            # Call NASA's JPL Horizons and compute orbital Earth's orbit at
            # observed times
            t_min = np.min([table["HJD"][0] for table in data.light_curves])
            t_max = np.max([table["HJD"][-1] for table in data.light_curves])

            t = np.linspace(t_min, t_max, 5000)
            times = Time(t, format="jd", scale="tdb")

            # Get Earth's position and velocity from JPL Horizons
            pos, vel = get_body_barycentric_posvel("earth", times)

            # Minus sign is because get_body_barycentric returns Earth position
            # in heliocentric coordinates
            s_t = -pos.xyz.value.T
            v_t = -vel.xyz.value.T

            # Acceleration is not provided by the astropy function so we compute
            # the derivative numerically
            a_t = np.gradient(v_t, axis=0)

            # Project vectors onto the plane of the sky
            zeta_e, zeta_n = self.__project_vector_on_sky(
                s_t, data.event_coordinates
            )
            zeta_e_dot, zeta_n_dot = self.__project_vector_on_sky(
                v_t, data.event_coordinates
            )
            zeta_e_ddot, zeta_n_ddot = self.__project_vector_on_sky(
                a_t, data.event_coordinates
            )

            # Initialize `exoplanet.interp.RegularGridInterpolator` objects
            # These interpolators are used to evaluate the zeta function and its
            # derivatives at arbitrary times which is needed to compute the
            # likelihood.
            points = [t - 2450000]  # Times used for interpolation

            # zeta(t) interpolators
            self.zeta_e_interp = xo.interp.RegularGridInterpolator(
                points, zeta_e[:, None]
            )
            self.zeta_n_interp = xo.interp.RegularGridInterpolator(
                points, zeta_n[:, None]
            )

            # dot[zeta(t)] interpolators
            self.zeta_e_dot_interp = xo.interp.RegularGridInterpolator(
                points, zeta_e_dot[:, None]
            )
            self.zeta_n_dot_interp = xo.interp.RegularGridInterpolator(
                points, zeta_n_dot[:, None]
            )

            # ddot[zeta(t)] interpolators
            self.zeta_e_ddot_interp = xo.interp.RegularGridInterpolator(
                points, zeta_e_ddot[:, None]
            )
            self.zeta_n_ddot_interp = xo.interp.RegularGridInterpolator(
                points, zeta_n_ddot[:, None]
            )

        # Convert ang. acceleration units from [yr]^{-2} to [d]^{-2}
        if (self.a_par and self.a_per) is not None:
            self.a_par /= 365.25 ** 2
            self.a_per /= 365.25 ** 2

    def __project_vector_on_sky(self, matrix, coordinates):
        """
        This function takes a 3D cartesian vector specified in the ICRS
        coordinate system and evaluated at differrent times
        :math:`(t_i,\dots, t_N)` and projects it onto a spherical coordinate
        system on the plane of the sky with the origin at the position
        defined by the coordinates.
        
        Parameters
        ----------
        matrix : ndarray
            Matrix of shape ``(len(times), 3)``
        coordinates : ``astropy.coordinates.SkyCoord``
            Coordinates on the sky.
        """
        # Unit vector normal to the plane of the sky in ICRS coordiantes
        direction = np.array(coordinates.cartesian.xyz.value)
        direction /= np.linalg.norm(direction)

        # Unit vector pointing north in ICRS coordinates
        e_north = np.array([0.0, 0.0, 1.0])

        # Sperical unit vectors of the coordinate system defined
        # on the plane of the sky which is perpendicular to the
        # source star direction
        e_east_sky = np.cross(e_north, direction)
        e_north_sky = np.cross(direction, e_east_sky)

        east_component = np.dot(matrix, e_east_sky)
        north_component = np.dot(matrix, e_north_sky)

        return east_component, north_component

    def __compute_delta_zeta(self, t):
        """
        Computes the components of delta_zeta vector - the deviation of the
        projected separation of the Sun relative to time :math:`t0`.

        Parameters
        ----------
        t : list
            List of theano tensors containing times for which
            :math:`\mathbf{u}(t)` is to be computed.
       
        Returns
        -------
        tuple
            North and East components of
            delta_zeta(t). 
        """
        # Lists with dimension n_bands
        zeta_e_list = []
        zeta_n_list = []

        for idx in range(len(self.data.light_curves)):
            # the interpolation function requires tensor of shape (n_points, 1)
            # as an input
            pts = tt.reshape(t[idx], (tt.shape(t[idx])[0], 1))
            zeta_e_list.append(self.zeta_e_interp.evaluate(pts).transpose())
            zeta_n_list.append(self.zeta_n_interp.evaluate(pts).transpose())

        #  Evalute zeta function and its derivative at t0
        zeta_e_t0 = self.zeta_e_interp.evaluate(tt.reshape(self.t0, (1, 1)))[
            0, 0
        ]
        zeta_n_t0 = self.zeta_n_interp.evaluate(tt.reshape(self.t0, (1, 1)))[
            0, 0
        ]
        zeta_e_dot_t0 = self.zeta_e_dot_interp.evaluate(
            tt.reshape(self.t0, (1, 1))
        )[0, 0]
        zeta_n_dot_t0 = self.zeta_n_dot_interp.evaluate(
            tt.reshape(self.t0, (1, 1))
        )[0, 0]

        # Compute the delta_zeta function
        delta_zeta_e = zeta_e_list - zeta_e_t0 - (t - self.t0) * zeta_e_dot_t0
        delta_zeta_n = zeta_n_list - zeta_n_t0 - (t - self.t0) * zeta_n_dot_t0

        return delta_zeta_n, delta_zeta_e

    def __compute_delta_zeta_dot(self, t):
        """
        Computes the first derivatives of the components of delta_zeta vector.

        Parameters
        ----------
        t : list
            List of theano.tensor containing times for which u(t) is to be
            computed. 
       
        Returns
        -------
        tuple
            North and East components of
            delta_zeta_dot(t). 
        """
        zeta_e_dot_list = []
        zeta_n_dot_list = []

        for idx in range(len(self.data.light_curves)):
            # the interpolation function requires tensor of shape (n_points, 1)
            # as an input
            pts = tt.reshape(t[idx], (tt.shape(t[idx])[0], 1))
            zeta_e_dot_list.append(
                self.zeta_e_dot_interp.evaluate(pts).transpose()
            )
            zeta_n_dot_list.append(
                self.zeta_n_dot_interp.evaluate(pts).transpose()
            )

        zeta_e_dot_t0 = self.zeta_e_dot_interp.evaluate(
            tt.reshape(self.t0, (1, 1))
        )[0, 0]
        zeta_n_dot_t0 = self.zeta_n_dot_interp.evaluate(
            tt.reshape(self.t0, (1, 1))
        )[0, 0]

        # Compute delta_zeta function
        delta_zeta_e_dot = zeta_e_dot_list - zeta_e_dot_t0
        delta_zeta_n_dot = zeta_n_dot_list - zeta_n_dot_t0

        return delta_zeta_n_dot, delta_zeta_e_dot

    def __compute_delta_zeta_ddot(self, t):
        """
        Computes the second derivatives of the components of delta_zeta vector.

        Parameters
        ----------
        t : list
            List of theano.tensor containing times for which u(t) is to be
            computed. 
       
        Returns
        -------
        tuple
            North and East components of delta_zeta_ddot(t). 
        """
        zeta_e_ddot_list = []
        zeta_n_ddot_list = []

        for idx in range(len(self.data.light_curves)):
            # the interpolation function requires tensor of shape (n_points, 1)
            # as an input
            pts = tt.reshape(t[idx], (tt.shape(t[idx])[0], 1))
            zeta_e_ddot_list.append(
                self.zeta_e_ddot_interp.evaluate(pts).transpose()
            )
            zeta_n_ddot_list.append(
                self.zeta_n_ddot_interp.evaluate(pts).transpose()
            )

        # Compute delta_zeta function
        delta_zeta_e_ddot = zeta_e_ddot_list
        delta_zeta_n_ddot = zeta_n_ddot_list

        return delta_zeta_n_ddot, delta_zeta_e_ddot

    def __compute_u(self, t, piEE, piEN):
        delta_zeta_n, delta_zeta_e = self.__compute_delta_zeta(t)

        u_per = self.u0 + piEN * delta_zeta_e - piEE * delta_zeta_n
        u_par = (
            (t - self.t0) / self.tE + piEE * delta_zeta_e + piEN * delta_zeta_n
        )

        return u_per, u_par

    def __compute_u_dot(self, t, piEE, piEN):
        delta_zeta_n_dot, delta_zeta_e_dot = self.__compute_delta_zeta_dot(t)

        u_per = piEN * delta_zeta_e_dot - piEE * delta_zeta_n_dot
        u_par = 1 / self.tE + piEE * delta_zeta_e_dot + piEN * delta_zeta_n_dot

        return u_per, u_par

    def compute_trajectory(self, t, return_components=False):
        """
        Computes the magnitude of the relative lens-source separation vector
        :math:`\mathbf{u}(t)`.         

        Parameters
        ----------
        t : list
            List of theano.tensor containing times for which u(t) is to be
            computed. 
        return_components : bool, optional
            If true, the function returns the tuple
            u, u_n, u_e, pi_E magnitude, by default False.

        Returns
        -------
        theano.tensor
            The magnitude of the separation vector :math:`\mathbf{u}(t)`
            for each time or a tuple with the two components of
            :math:`\mathbf{u}(t)` in (north, east) basis.
        """

        if (self.piEE and self.piEN) is not None:
            u_per, u_par = self.__compute_u(t, self.piEE, self.piEN)

            u = tt.sqrt(u_per ** 2 + u_par ** 2)

            if return_components is True:
                piE = tt.sqrt(self.piEE ** 2 + self.piEN ** 2)

                cospsi = self.piEN / piE
                sinpsi = self.piEE / piE

                u_e = cospsi * u_per + sinpsi * u_par
                u_n = -sinpsi * u_per + cospsi * u_par

                return u, u_n, u_e

            else:
                return u

        elif (self.piE and self.psi) is not None:
            cospsi = tt.cos(self.psi)
            sinpsi = tt.sin(self.psi)

            piEE = self.piE * sinpsi
            piEN = self.piE * cospsi

            u_per, u_par = self.__compute_u(t, piEE, piEN)

            u = tt.sqrt(u_per ** 2 + u_par ** 2)

            if return_components is True:
                u_e = cospsi * u_per + sinpsi * u_par
                u_n = -sinpsi * u_per + cospsi * u_par

                return u, u_n, u_e

            else:
                return u

        elif (self.a_par and self.a_per) is not None:
            delta_zeta_n_ddot_t0, delta_zeta_e_ddot_t0 = self.__compute_delta_zeta_ddot(
                [self.t0.reshape([1, 1])]
            )

            delta_zeta_e_ddot_t0 = delta_zeta_e_ddot_t0[0][0][0]
            delta_zeta_n_ddot_t0 = delta_zeta_n_ddot_t0[0][0][0]

            piEN = (
                self.a_par * delta_zeta_n_ddot_t0
                + self.a_per * delta_zeta_e_ddot_t0
            ) / (delta_zeta_e_ddot_t0 ** 2 + delta_zeta_n_ddot_t0 ** 2)

            piEE = (
                self.a_par * delta_zeta_e_ddot_t0
                - self.a_per * delta_zeta_n_ddot_t0
            ) / (delta_zeta_e_ddot_t0 ** 2 + delta_zeta_n_ddot_t0 ** 2)

            piE = tt.sqrt(piEE ** 2 + piEN ** 2)

            u_per, u_par = self.__compute_u(t, piEE, piEN)

            u = tt.sqrt(u_per ** 2 + u_par ** 2)

            if return_components is True:
                cospsi = piEN / piE
                sinpsi = piEE / piE

                u_e = cospsi * u_per + sinpsi * u_par
                u_n = -sinpsi * u_per + cospsi * u_par

                return u, u_n, u_e, piE

            else:
                return u
        else:
            u = tt.sqrt(self.u0 ** 2 + ((t - self.t0) / self.tE) ** 2)
            return u
