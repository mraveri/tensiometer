"""Tests for CPCA and KL utilities in flow_CPCA."""

#########################################################################################################
# Imports

import types
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from tensiometer.synthetic_probability import flow_CPCA

#########################################################################################################
# Test configuration

tf.config.run_functions_eagerly(True)

#########################################################################################################
# Helper classes


class DummyFlow:
    """Minimal flow object exposing the hooks expected by flow_CPCA helpers."""

    def __init__(self):
        """Init."""
        self.num_params = 2

    def cast(self, v):
        """Cast."""
        return tf.cast(v, tf.float64)

    def _batch_eye(self, x, scale=1.0):
        """Batch eye."""
        batch = tf.shape(tf.convert_to_tensor(x))[0]
        eye = tf.eye(self.num_params, dtype=tf.float64) * scale
        return tf.tile(tf.expand_dims(eye, 0), [batch, 1, 1])

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return self.cast(x)

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return self.cast(x)

    def inverse_jacobian(self, x):
        """Inverse jacobian."""
        return self._batch_eye(x)

    def direct_jacobian(self, x):
        """Direct jacobian."""
        return self._batch_eye(x)

    def metric(self, x):
        """Metric."""
        return self._batch_eye(x)

#########################################################################################################
# CPCA tests


class TestFlowCpca(unittest.TestCase):
    """Flow CPCA test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.flow = DummyFlow()
        self.prior_flow = DummyFlow()
        # bind helpers that are defined as free functions in the module
        self.flow._naive_eigenvalue_ode_abs = types.MethodType(
            flow_CPCA._naive_eigenvalue_ode_abs, self.flow
        )
        self.flow.solve_eigenvalue_ode_abs = types.MethodType(
            flow_CPCA.solve_eigenvalue_ode_abs, self.flow
        )
        self.flow.solve_eigenvalue_ode_par = types.MethodType(
            flow_CPCA.solve_eigenvalue_ode_par, self.flow
        )
        self.flow.eigenvalue_ode_abs_temp_3 = types.MethodType(
            flow_CPCA.eigenvalue_ode_abs_temp_3, self.flow
        )

    def test_tf_cpc_decomposition(self):
        """Test TF CPC decomposition."""
        a = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float64)
        b = tf.eye(2, dtype=tf.float64)
        eig, eigv = flow_CPCA.tf_CPC_decomposition(a, b)
        np.testing.assert_allclose(np.sort(eig.numpy()), [1.0, 2.0])
        self.assertEqual(eigv.shape, (2, 2))

    def test_naive_eigenvalue_ode_and_solver(self):
        """Test naive eigenvalue ODE and solver."""
        reference = np.array([1.0, 0.0])
        direction = self.flow._naive_eigenvalue_ode_abs(0.0, np.zeros(2), reference)
        self.assertEqual(direction.shape, (1, 2))
        times, traj, vel = self.flow.solve_eigenvalue_ode_abs(
            np.zeros(2), n=0, num_points=3
        )
        self.assertEqual(times.shape[0], 5)
        self.assertEqual(traj.shape[1], self.flow.num_params)
        self.assertEqual(vel.shape[1], self.flow.num_params)

    def test_eigenvalue_ode_in_parameter_space(self):
        """Test eigenvalue ODE in parameter space."""
        times, traj = self.flow.solve_eigenvalue_ode_par(
            np.array([0.2, -0.1]), n=1, num_points=3
        )
        self.assertEqual(times.shape[0], 5)
        self.assertEqual(traj.shape[1], self.flow.num_params)

    def test_kl_ode_and_solver(self):
        """Test KL ODE and solver."""
        reference = np.array([0.0, 1.0])
        w = flow_CPCA._naive_KL_ode(0.0, np.zeros(2), reference, self.flow, self.prior_flow)
        self.assertEqual(w.shape, (1, 2))
        times, traj, vel = flow_CPCA.solve_KL_ode(
            self.flow, self.prior_flow, np.zeros(2),
            n=0, length=0.2, num_points=3
        )
        self.assertEqual(times.shape[0], 5)
        self.assertEqual(traj.shape[1], self.flow.num_params)
        self.assertEqual(vel.shape[1], self.flow.num_params)

    def test_eigenvalue_ode_abs_temp_placeholder(self):
        """Test eigenvalue ODE abs temp placeholder."""
        y = tf.constant([0.0, 0.0, 1.0, 0.0, 0.0], dtype=tf.float64)
        update = self.flow.eigenvalue_ode_abs_temp_3(0.0, y)
        self.assertEqual(update.shape, (5,))

#########################################################################################################
# ODE fakes


class FakeOde:
    """Simple stand-in for scipy.integrate.ode."""

    def __init__(self, func, raise_once=False):
        """Init."""
        self.func = func
        self.raise_once = raise_once
        self.y = None
        self.params = ()

    def set_integrator(self, **kwargs):
        """Set integrator."""
        return self

    def set_initial_value(self, y0, t0):
        """Set initial value."""
        self.y = np.array(y0, dtype=float)
        return self

    def set_f_params(self, *args):
        """Set f params."""
        self.params = args
        return self

    def integrate(self, t):
        """Integrate."""
        if self.raise_once:
            self.raise_once = False
            raise RuntimeError("integration failed")
        return self.y


#########################################################################################################
# Additional helper classes


class DummyFlowAdditional:
    """Minimal flow mock providing the interfaces required by flow_CPCA."""

    def __init__(self, num_params=2):
        """Init."""
        self.num_params = num_params
        # bind module functions as methods
        self._naive_eigenvalue_ode_abs = flow_CPCA._naive_eigenvalue_ode_abs.__get__(self, DummyFlowAdditional)
        self.solve_eigenvalue_ode_abs = flow_CPCA.solve_eigenvalue_ode_abs.__get__(self, DummyFlowAdditional)
        self.solve_eigenvalue_ode_par = flow_CPCA.solve_eigenvalue_ode_par.__get__(self, DummyFlowAdditional)
        self.eigenvalue_ode_abs_temp_3 = flow_CPCA.eigenvalue_ode_abs_temp_3.__get__(self, DummyFlowAdditional)

    def cast(self, arr):
        """Cast."""
        return tf.convert_to_tensor(arr, dtype=tf.float64)

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return x

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return x

    def inverse_jacobian(self, x):
        """Inverse jacobian."""
        return [tf.eye(self.num_params, dtype=tf.float64)]

    def direct_jacobian(self, x):
        """Direct jacobian."""
        return [tf.eye(self.num_params, dtype=tf.float64)]

    def metric(self, x):
        """Metric."""
        return [tf.eye(self.num_params, dtype=tf.float64)]


#########################################################################################################
# Additional CPCA tests


class TestFlowCpcaAdditional(unittest.TestCase):
    """Flow CPCA additional test suite."""
    def test_tf_decompositions(self):
        """Test TF decompositions."""
        a = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float64)
        b = tf.eye(2, dtype=tf.float64)
        eig, vec = flow_CPCA.tf_CPC_decomposition(a, b)
        self.assertEqual(eig.shape[0], 2)
        eig2, vec2 = flow_CPCA.tf_KL_decomposition(a, b)
        self.assertTrue(tf.reduce_all(eig == eig2))
        self.assertEqual(vec.shape, vec2.shape)

    def test_eigenvalue_ode_abs_branches(self):
        """Test eigenvalue ODE abs branches."""
        flow = DummyFlowAdditional(num_params=2)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=True)):
            times, traj, vel = flow.solve_eigenvalue_ode_abs([0.0, 0.0], 0, side='+',
                                                             integrator_options={"name": "dummy"},
                                                             num_points=3)
        self.assertEqual(traj.shape[0], 3)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=True)):
            times_b, traj_b, vel_b = flow.solve_eigenvalue_ode_abs([0.0, 0.0], 0, side='-',
                                                                   integrator_options={"name": "dummy"},
                                                                   num_points=3)
        self.assertEqual(traj_b.shape[0], 3)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=False)):
            times_c, traj_c, vel_c = flow.solve_eigenvalue_ode_abs([0.0, 0.0], 0, side='both',
                                                                   num_points=3)
        self.assertEqual(traj_c.shape[0], 5)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=False)):
            with self.assertRaises(UnboundLocalError):
                flow.solve_eigenvalue_ode_abs([0.0, 0.0], 0, side='invalid', num_points=3)

    def test_eigenvalue_ode_par_and_temp3(self):
        """Test eigenvalue ODE parameter and temp3 helpers."""
        flow = DummyFlowAdditional(num_params=2)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=False)):
            times, traj = flow.solve_eigenvalue_ode_par([0.0, 0.0], 0, side='+', num_points=3)
        self.assertEqual(traj.shape[1], 2)
        y_full = tf.constant([0.0, 0.0, 1.0, 0.5, 0.0], dtype=tf.float64)
        out = flow.eigenvalue_ode_abs_temp_3(0.0, y_full)
        self.assertEqual(out.shape[0], 5)

    def test_kl_ode_branches(self):
        """Test KL ODE branches."""
        flow = DummyFlowAdditional(num_params=2)
        prior = DummyFlowAdditional(num_params=2)
        w = flow_CPCA._naive_KL_ode(0.0, np.array([0.1, 0.2]), np.array([1.0, 0.0]), flow, prior)
        self.assertEqual(w.shape[1], 2)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=True)):
            times, traj, vel = flow_CPCA.solve_KL_ode(flow, prior, [0.0, 0.0], 0, side='+',
                                                      integrator_options={"name": "dummy"},
                                                      num_points=3)
        self.assertEqual(traj.shape[0], 3)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=True)):
            times_b, traj_b, vel_b = flow_CPCA.solve_KL_ode(flow, prior, [0.0, 0.0], 0, side='-',
                                                            integrator_options={"name": "dummy"},
                                                            num_points=3)
            times_c, traj_c, vel_c = flow_CPCA.solve_KL_ode(flow, prior, [0.0, 0.0], 0, side='both',
                                                            num_points=3)
        self.assertEqual(traj_b.shape[0], 3)
        self.assertEqual(traj_c.shape[0], 5)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=False)):
            with self.assertRaises(UnboundLocalError):
                flow_CPCA.solve_KL_ode(flow, prior, [0.0, 0.0], 0, side='invalid', num_points=3)


#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
