import numpy as np
import theano.tensor as T

from caustic import data, models, trajectory


def test_delta_zeta():
    """
    By definition, delta_zeta(t=t0)=0.
    """
    # Data is used just to initialize the parallax model
    event = data.OGLEData("data/OB05086")

    model = models.SingleLensModel(event)

    with model:
        trajectory_ = trajectory.Trajectory(event, 50.0, 0.1, 20.0, 0.1, 0.3)

        t0_test = T.as_tensor_variable(np.array([50.0])[np.newaxis, :])

        delta_zeta_n, delta_zeta_e = trajectory_._Trajectory__compute_delta_zeta(
            t0_test
        )

        assert delta_zeta_e.eval()[0][0][0] == 0.0
        assert delta_zeta_n.eval()[0][0][0] == 0.0


def test_delta_zeta_dot():
    """
    By definition, delta_zeta_dot(t=t0)=0.
    """
    # Data is used just to initialize the parallax model
    event = data.OGLEData("data/OB05086")

    model = models.SingleLensModel(event)

    with model:
        trajectory_ = trajectory.Trajectory(event, 50.0, 0.1, 20.0, 0.1, 0.3)

        t0_test = T.as_tensor_variable(np.array([50.0])[np.newaxis, :])

        delta_zeta_n_dot, delta_zeta_e_dot = trajectory_._Trajectory__compute_delta_zeta_dot(
            t0_test
        )

        assert delta_zeta_e_dot.eval()[0][0][0] == 0.0
        assert delta_zeta_n_dot.eval()[0][0][0] == 0.0


def test_compute_trajectory():
    """
    At t0, u(t) and u_dot(t) are perpendicular. Also for piEE=piEN=0 the model
    should reduce to one without a parallax, same goes fore a_per=a_par=0.
    """
    # Data is used just to initialize the parallax model
    event = data.OGLEData("data/OB05086")

    # Check that u is perpendicular t0 u_dot at t0
    model = models.SingleLensModel(event)

    with model:
        t0_ = 0.5 * (model.t[0][0].eval() + model.t[0][-1].eval())

        trajectory_ = trajectory.Trajectory(
            event, t0_, 0.4, 150.0, piEE=0.1, piEN=0.3
        )
        t0_test = T.as_tensor_variable(np.array([t0_])[np.newaxis, :])

        un, ue = trajectory_._Trajectory__compute_u(t0_test, 0.1, 0.3)
        un_dot, ue_dot = trajectory_._Trajectory__compute_u_dot(
            t0_test, 0.1, 0.3
        )

        u = T.stack([ue[0][0][0], un[0][0][0]]).T
        u_dot = T.stack([ue_dot[0][0][0], un_dot[0][0][0]])
        dot = T.dot(u, u_dot)

        assert np.allclose(dot.eval(), 0.0)

    # Check that all parallax models reduce to the standard model when the
    # parallax parameters are zero
    model = models.SingleLensModel(event)

    with model:
        t0_ = 0.5 * (model.t[0][0].eval() + model.t[0][-1].eval())

        trajectory_1 = trajectory.Trajectory(
            event,
            T.as_tensor_variable(t0_),
            T.as_tensor_variable(0.4),
            T.as_tensor_variable(150.0),
        )

        trajectory_2 = trajectory.Trajectory(
            event,
            T.as_tensor_variable(t0_),
            T.as_tensor_variable(0.4),
            T.as_tensor_variable(150.0),
            piEE=T.as_tensor_variable(0.0),
            piEN=T.as_tensor_variable(0.0),
        )

        trajectory_3 = trajectory.Trajectory(
            event,
            T.as_tensor_variable(t0_),
            T.as_tensor_variable(0.4),
            T.as_tensor_variable(150.0),
            a_per=T.as_tensor_variable(0.0),
            a_par=T.as_tensor_variable(0.0),
        )

        u1 = trajectory_1.compute_trajectory(model.t)
        u2 = trajectory_2.compute_trajectory(model.t)
        u3 = trajectory_3.compute_trajectory(model.t)

        assert np.allclose(u1.eval(), u2.eval())
        assert np.allclose(u2.eval(), u3.eval())

    # Check consistency between the three parametrizations of the parallax model
    model = models.SingleLensModel(event)

    with model:
        t0_ = 0.5 * (model.t[0][0].eval() + model.t[0][-1].eval())
        t0_tens = T.as_tensor_variable(np.array([t0_])[np.newaxis, :])

        # Specify parallax parameters
        piEE = 0.1
        piEN = 0.3

        # Get corresponding parallax params in the angle parametrization
        piE_ = np.sqrt(piEE ** 2 + piEN ** 2)
        sinpsi = piEE / piE_
        cospsi = piEN / piE_

        psi = np.arctan2(sinpsi, cospsi)

        # Same for acceleration parametrization
        trajectory_1 = trajectory.Trajectory(
            event,
            T.as_tensor_variable(t0_),
            T.as_tensor_variable(0.4),
            T.as_tensor_variable(150.0),
            piEE=T.as_tensor_variable(piEE),
            piEN=T.as_tensor_variable(piEN),
        )

        delta_zeta_n_ddot_t0, delta_zeta_e_ddot_t0 = trajectory_1._Trajectory__compute_delta_zeta_ddot(
            t0_tens
        )
        delta_zeta_n_ddot_t0 = delta_zeta_n_ddot_t0[0][0][0]
        delta_zeta_e_ddot_t0 = delta_zeta_e_ddot_t0[0][0][0]

        a_per = piEN * delta_zeta_e_ddot_t0 - piEE * delta_zeta_n_ddot_t0
        a_par = piEN * delta_zeta_n_ddot_t0 + piEE * delta_zeta_e_ddot_t0

        # Convert units
        a_per *= 365.25 ** 2
        a_par *= 365.25 ** 2

        trajectory_2 = trajectory.Trajectory(
            event,
            T.as_tensor_variable(t0_),
            T.as_tensor_variable(0.4),
            T.as_tensor_variable(150.0),
            piE=T.as_tensor_variable(piE_),
            psi=T.as_tensor_variable(psi),
        )

        trajectory_3 = trajectory.Trajectory(
            event,
            T.as_tensor_variable(t0_),
            T.as_tensor_variable(0.4),
            T.as_tensor_variable(150.0),
            a_per=T.as_tensor_variable(a_per),
            a_par=T.as_tensor_variable(a_par),
        )

        u1, u1n, u1e = trajectory_1.compute_trajectory(
            model.t, return_components=True
        )
        u2, u2n, u2e = trajectory_2.compute_trajectory(
            model.t, return_components=True
        )
        u3, u3n, u3e, piE = trajectory_3.compute_trajectory(
            model.t, return_components=True
        )

        assert np.allclose(u1.eval(), u2.eval())
        assert np.allclose(u2.eval(), u3.eval())
        assert np.allclose(u1n.eval(), u2n.eval())
        assert np.allclose(u2n.eval(), u3n.eval())
        assert np.allclose(u1e.eval(), u2e.eval())
        assert np.allclose(u2e.eval(), u3e.eval())
        assert np.allclose(piE.eval(), np.sqrt(piEE ** 2 + piEN ** 2))
