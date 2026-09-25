"""Tests for CPCA and KL utilities in flow_CPCA."""

#########################################################################################################
# Imports

import types
import unittest
from unittest.mock import patch

import numpy as np
import scipy.linalg
import torch

from tensiometer.synthetic_probability import flow_CPCA
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Test configuration

DOPRI5 = {'name': 'dopri5'}

#########################################################################################################
# Helper functions


def _tolerance():
    """Return a relative tolerance suited to the active precision."""
    if tu.get_precision() == torch.float64:
        return 1e-10
    return 1e-4


def _random_spd(rng, dim):
    """Return a random well conditioned symmetric positive definite matrix.

    :param rng: numpy random generator.
    :param dim: matrix dimension.
    :returns: matrix with shape ``(dim, dim)``.
    """
    temp = rng.normal(size=(dim, dim))
    return temp @ temp.T + dim * np.eye(dim)

def _check_straight_line(test, times, traj, direction):
    """Check that ``traj`` is the line ``times * e_direction`` (up to an overall sign).

    :param test: test case used for the assertions.
    :param times: solution times, shape ``(N,)``.
    :param traj: trajectory, shape ``(N, D)``.
    :param direction: index of the unit vector followed by the trajectory.
    """
    traj = np.asarray(traj, dtype=np.float64)
    expected = np.zeros_like(traj)
    expected[:, direction] = times
    sign = 1.0 if np.sum(traj * expected) >= 0.0 else -1.0
    np.testing.assert_allclose(sign * traj, expected, atol=1e-5)

#########################################################################################################
# Helper classes


class DummyFlow:
    """Minimal flow object exposing the hooks expected by flow_CPCA helpers."""

    def __init__(self):
        """Init."""
        self.num_params = 2

    def cast(self, v):
        """Cast to a CPU tensor with the active precision."""
        return tu.to_tensor(v, device='cpu')

    def _batch_eye(self, x, scale=1.0):
        """Batch of identity matrices with the batch size of ``x``."""
        batch = self.cast(x).shape[0]
        eye = torch.eye(self.num_params, dtype=tu.get_precision()) * scale
        return eye.expand(batch, self.num_params, self.num_params)

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

    def test_torch_cpc_decomposition(self):
        """Test torch CPC decomposition on a diagonal pair."""
        a = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=tu.get_precision())
        b = torch.eye(2, dtype=tu.get_precision())
        eig, eigv = flow_CPCA.torch_CPC_decomposition(a, b)
        np.testing.assert_allclose(np.sort(tu.to_numpy(eig)), [1.0, 2.0])
        self.assertEqual(tuple(eigv.shape), (2, 2))
        self.assertEqual(eig.dtype, tu.get_precision())

    def test_torch_cpc_matches_scipy_generalized_eigenproblem(self):
        """Test torch CPC decomposition against scipy on a random SPD pair."""
        rng = np.random.default_rng(1)
        dim = 4
        matrix_a = _random_spd(rng, dim)
        matrix_b = _random_spd(rng, dim)
        eig, eigv = flow_CPCA.torch_CPC_decomposition(
            tu.to_tensor(matrix_a, device='cpu'), tu.to_tensor(matrix_b, device='cpu'))
        eig = tu.to_numpy(eig).astype(np.float64)
        eigv = tu.to_numpy(eigv).astype(np.float64)
        ref_eig, ref_eigv = scipy.linalg.eigh(matrix_a, matrix_b)
        rtol = _tolerance()
        # eigenvalues in ascending order, as in scipy:
        np.testing.assert_allclose(eig, ref_eig, rtol=rtol)
        # generalized eigenproblem A v = lambda B v:
        np.testing.assert_allclose(matrix_a @ eigv, matrix_b @ eigv * eig[None, :],
                                   rtol=rtol, atol=rtol * np.abs(matrix_a).max())
        # B-orthonormal eigenvectors:
        np.testing.assert_allclose(eigv.T @ matrix_b @ eigv, np.eye(dim), atol=10 * rtol)
        # same eigenvectors as scipy up to a sign:
        signs = np.sign(np.sum(eigv * ref_eigv, axis=0))
        np.testing.assert_allclose(eigv * signs[None, :], ref_eigv, rtol=rtol, atol=10 * rtol)

    def test_torch_cpc_batched(self):
        """Test torch CPC decomposition on a batch of SPD pairs."""
        rng = np.random.default_rng(2)
        dim = 3
        matrices_a = np.stack([_random_spd(rng, dim) for _ in range(2)])
        matrices_b = np.stack([_random_spd(rng, dim) for _ in range(2)])
        eig, eigv = flow_CPCA.torch_CPC_decomposition(
            tu.to_tensor(matrices_a, device='cpu'), tu.to_tensor(matrices_b, device='cpu'))
        self.assertEqual(tuple(eig.shape), (2, dim))
        self.assertEqual(tuple(eigv.shape), (2, dim, dim))
        for ind in range(2):
            ref_eig = scipy.linalg.eigh(matrices_a[ind], matrices_b[ind], eigvals_only=True)
            np.testing.assert_allclose(tu.to_numpy(eig[ind]), ref_eig, rtol=_tolerance())

    def test_naive_eigenvalue_ode_and_solver(self):
        """Test naive eigenvalue ODE and solver."""
        reference = np.array([1.0, 0.0])
        direction = self.flow._naive_eigenvalue_ode_abs(0.0, np.zeros(2), reference)
        self.assertEqual(tuple(direction.shape), (1, 2))
        times, traj, vel = self.flow.solve_eigenvalue_ode_abs(
            np.zeros(2), n=0, num_points=3, integrator_options=DOPRI5
        )
        self.assertIsInstance(times, np.ndarray)
        self.assertEqual(times.shape[0], 5)
        self.assertEqual(traj.shape[1], self.flow.num_params)
        self.assertEqual(vel.shape[1], self.flow.num_params)
        # with an identity Jacobian the path is a straight line along the first eigenvector:
        _check_straight_line(self, times, traj, direction=0)

    def test_eigenvalue_ode_with_default_integrator(self):
        """Test the eigenvalue ODE with the default scipy integrator (vode needs a flat right hand side)."""
        times, traj, vel = self.flow.solve_eigenvalue_ode_abs(
            np.zeros(2), n=0, side='+', num_points=3
        )
        _check_straight_line(self, times, traj, direction=0)

    def test_eigenvalue_ode_in_parameter_space(self):
        """Test eigenvalue ODE in parameter space."""
        start = np.array([0.2, -0.1])
        times, traj = self.flow.solve_eigenvalue_ode_par(
            start, n=1, num_points=3, integrator_options=DOPRI5
        )
        self.assertEqual(times.shape[0], 5)
        self.assertEqual(traj.shape[1], self.flow.num_params)
        _check_straight_line(self, times, tu.to_numpy(traj) - start, direction=1)

    def test_kl_ode_and_solver(self):
        """Test KL ODE and solver."""
        reference = np.array([0.0, 1.0])
        w = flow_CPCA._naive_KL_ode(0.0, np.zeros(2), reference, self.flow, self.prior_flow)
        self.assertEqual(tuple(w.shape), (1, 2))
        times, traj, vel = flow_CPCA.solve_KL_ode(
            self.flow, self.prior_flow, np.zeros(2),
            n=0, length=0.2, num_points=3, integrator_options=DOPRI5
        )
        self.assertIsInstance(times, np.ndarray)
        self.assertEqual(times.shape[0], 5)
        self.assertEqual(traj.shape[1], self.flow.num_params)
        self.assertEqual(vel.shape[1], self.flow.num_params)
        _check_straight_line(self, times, traj, direction=0)

    def test_eigenvalue_ode_abs_temp_placeholder(self):
        """Test eigenvalue ODE abs temp placeholder."""
        y = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=tu.get_precision())
        update = self.flow.eigenvalue_ode_abs_temp_3(0.0, y)
        self.assertEqual(tuple(update.shape), (5,))

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
        """Cast to a CPU tensor with the active precision."""
        return tu.to_tensor(arr, device='cpu')

    def _eye(self):
        """Identity matrix with the active precision."""
        return torch.eye(self.num_params, dtype=tu.get_precision())

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return x

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return x

    def inverse_jacobian(self, x):
        """Inverse jacobian."""
        return [self._eye()]

    def direct_jacobian(self, x):
        """Direct jacobian."""
        return [self._eye()]

    def metric(self, x):
        """Metric."""
        return [self._eye()]


#########################################################################################################
# Additional CPCA tests


class TestFlowCpcaAdditional(unittest.TestCase):
    """Flow CPCA additional test suite."""
    def test_torch_decompositions(self):
        """Test that the KL decomposition matches the CPC decomposition."""
        a = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=tu.get_precision())
        b = torch.eye(2, dtype=tu.get_precision())
        eig, vec = flow_CPCA.torch_CPC_decomposition(a, b)
        self.assertEqual(eig.shape[0], 2)
        eig2, vec2 = flow_CPCA.torch_KL_decomposition(a, b)
        self.assertTrue(torch.equal(eig, eig2))
        self.assertEqual(vec.shape, vec2.shape)

    def test_decomposition_accepts_numpy(self):
        """Test that numpy inputs are converted to tensors."""
        eig, vec = flow_CPCA.torch_CPC_decomposition(np.diag([3.0, 1.0]), np.eye(2))
        self.assertTrue(torch.is_tensor(eig))
        np.testing.assert_allclose(tu.to_numpy(eig), [1.0, 3.0])

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
        np.testing.assert_allclose(times_b, -np.linspace(0., 1.5, 3))
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=False)):
            times_c, traj_c, vel_c = flow.solve_eigenvalue_ode_abs([0.0, 0.0], 0, side='both',
                                                                   num_points=3)
        self.assertEqual(traj_c.shape[0], 5)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=False)):
            with self.assertRaises(ValueError):
                flow.solve_eigenvalue_ode_abs([0.0, 0.0], 0, side='invalid', num_points=3)

    def test_eigenvalue_ode_par_and_temp3(self):
        """Test eigenvalue ODE parameter and temp3 helpers."""
        flow = DummyFlowAdditional(num_params=2)
        with patch("tensiometer.synthetic_probability.flow_CPCA.scipy.integrate.ode",
                   lambda func: FakeOde(func, raise_once=False)):
            times, traj = flow.solve_eigenvalue_ode_par([0.0, 0.0], 0, side='+', num_points=3)
        self.assertEqual(traj.shape[1], 2)
        y_full = torch.tensor([0.0, 0.0, 1.0, 0.5, 0.0], dtype=tu.get_precision())
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
            with self.assertRaises(ValueError):
                flow_CPCA.solve_KL_ode(flow, prior, [0.0, 0.0], 0, side='invalid', num_points=3)


#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
