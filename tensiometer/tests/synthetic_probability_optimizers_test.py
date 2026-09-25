"""Tests for the batched BFGS minimizer and the torch minimization helpers of the flow profiler."""

#########################################################################################################
# Imports

import unittest

import numpy as np
import scipy.optimize
import torch

import tensiometer.synthetic_probability.flow_profiler as fp
from tensiometer.synthetic_probability import optimizers
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Helpers


def _gradient_tolerance():
    """Gradient tolerance that the active precision can reach."""
    if tu.get_precision() == torch.float64:
        return 1e-10
    return 1e-4


def _quadratic_problem(centres, hessians):
    """Batch of shifted quadratics ``0.5 (x - c_b)^T A_b (x - c_b)``.

    :param centres: minima, shape ``(B, d)``
    :param hessians: Hessians, shape ``(d, d)`` or ``(B, d, d)``
    :returns: callable mapping ``(B, d)`` positions to values ``(B,)`` and gradients ``(B, d)``
    """

    def value_and_grad(x):
        """Values and gradients of the quadratics."""
        centre = torch.as_tensor(centres, dtype=x.dtype, device=x.device)
        hessian = torch.broadcast_to(torch.as_tensor(hessians, dtype=x.dtype, device=x.device),
                                     (x.shape[0], x.shape[1], x.shape[1]))
        diff = x - centre
        grad = torch.einsum("bij,bj->bi", hessian, diff)
        return 0.5 * (diff * grad).sum(-1), grad

    return value_and_grad


def _rosenbrock_numpy(x, a):
    """Rosenbrock function with minimum at ``(a, a^2)``."""
    return (a - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2


def _rosenbrock_numpy_grad(x, a):
    """Gradient of :func:`_rosenbrock_numpy`."""
    return np.array([-2.0 * (a - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2), 200.0 * (x[1] - x[0]**2)])


def _rosenbrock_problem(a):
    """Batch of Rosenbrock functions with row dependent minima ``(a_b, a_b^2)``."""

    def value_and_grad(x):
        """Values and gradients of the Rosenbrock functions."""
        shift = torch.as_tensor(a, dtype=x.dtype, device=x.device)
        x0, x1 = x[:, 0], x[:, 1]
        value = (shift - x0)**2 + 100.0 * (x1 - x0**2)**2
        grad = torch.stack([-2.0 * (shift - x0) - 400.0 * x0 * (x1 - x0**2), 200.0 * (x1 - x0**2)], dim=-1)
        return value, grad

    return value_and_grad


class GaussianStubFlow:
    """Flow stub with a Gaussian log probability."""

    def __init__(self, mean, precision_matrix):
        """Build the stub.

        :param mean: mean, shape ``(D,)``
        :param precision_matrix: inverse covariance, shape ``(D, D)``
        """
        self.mean = np.asarray(mean, dtype=np.float64)
        self.precision_matrix = np.asarray(precision_matrix, dtype=np.float64)
        self.num_params = len(mean)
        self.param_names = ["p" + str(i) for i in range(self.num_params)]
        self.param_labels = ["P" + str(i) for i in range(self.num_params)]
        self.parameter_ranges = None
        self.name_tag = "gaussian"
        rng = np.random.default_rng(0)
        self.chain_samples = (self.mean + 0.1 * rng.standard_normal((50, self.num_params))).astype(tu.np_prec)
        self.chain_loglikes = np.zeros(50, dtype=tu.np_prec)

    def cast(self, arr):
        """Convert to a CPU tensor in the active precision."""
        return tu.to_tensor(arr, device="cpu")

    def log_probability(self, x):
        """Unnormalized Gaussian log probability."""
        x = tu.to_tensor(x)
        diff = x - tu.to_tensor(self.mean, device=x.device)
        precision_matrix = tu.to_tensor(self.precision_matrix, device=x.device)
        return -0.5 * torch.einsum("ni,ij,nj->n", diff, precision_matrix, diff)

    def log_probability_jacobian(self, x):
        """Gradient of the Gaussian log probability."""
        x = tu.to_tensor(x)
        diff = x - tu.to_tensor(self.mean, device=x.device)
        return -diff @ tu.to_tensor(self.precision_matrix, device=x.device)

#########################################################################################################
# batched_minimize


class TestBatchedMinimize(unittest.TestCase):
    """Convergence, options and failure handling of batched_minimize."""

    def setUp(self):
        """Seed the random generators."""
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)

    def test_shifted_quadratics_converge(self):
        """Independent shifted quadratics converge to their centres."""
        centres = self.rng.normal(size=(5, 3))
        hessian = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.0]])
        initial = torch.zeros(5, 3, dtype=tu.get_precision())
        tolerance = _gradient_tolerance()
        result = optimizers.batched_minimize(_quadratic_problem(centres, hessian), initial,
                                             tolerance=tolerance, max_iterations=100)
        self.assertEqual(result.position.dtype, tu.get_precision())
        self.assertEqual(result.position.device, initial.device)
        self.assertEqual(result.position.shape, (5, 3))
        self.assertEqual(result.objective_value.shape, (5,))
        self.assertEqual(result.objective_gradient.shape, (5, 3))
        self.assertEqual(result.inverse_hessian_estimate.shape, (5, 3, 3))
        self.assertTrue(bool(result.converged.all()))
        self.assertFalse(bool(result.failed.any()))
        self.assertLessEqual(float(result.objective_gradient.abs().max()), tolerance)
        self.assertTrue(np.allclose(tu.to_numpy(result.position), centres, atol=10 * tolerance))
        # the BFGS estimates stay symmetric positive definite:
        estimates = tu.to_numpy(result.inverse_hessian_estimate).astype(np.float64)
        self.assertTrue(np.allclose(estimates, np.transpose(estimates, (0, 2, 1)), atol=1e-4))
        self.assertTrue(np.all(np.linalg.eigvalsh(estimates) > 0.0))

    def test_one_dimensional_input(self):
        """A single point of shape ``(d,)`` is treated as a batch of one."""
        result = optimizers.batched_minimize(_quadratic_problem(np.ones((1, 2)), np.eye(2)),
                                             torch.zeros(2, dtype=tu.get_precision()), tolerance=_gradient_tolerance())
        self.assertEqual(result.position.shape, (1, 2))
        self.assertTrue(bool(result.converged.all()))

    def test_already_converged(self):
        """Points at the minimum converge without iterating."""
        result = optimizers.batched_minimize(_quadratic_problem(np.zeros((2, 2)), np.eye(2)),
                                             torch.zeros(2, 2, dtype=tu.get_precision()))
        self.assertEqual(result.num_iterations, 0)
        self.assertTrue(bool(result.converged.all()))

    def test_rosenbrock_matches_scipy(self):
        """A float64 Rosenbrock batch matches scipy BFGS."""
        shifts = np.array([1.0, 0.5, -0.8, 1.5])
        initial = np.array([[-1.2, 1.0], [0.0, 0.0], [0.5, 1.5], [2.0, 2.0]])
        result = optimizers.batched_minimize(_rosenbrock_problem(shifts), torch.tensor(initial, dtype=torch.float64),
                                             tolerance=1e-8, max_iterations=1000)
        self.assertEqual(result.position.dtype, torch.float64)
        self.assertTrue(bool(result.converged.all()))
        self.assertFalse(bool(result.failed.any()))
        for i, shift in enumerate(shifts):
            reference = scipy.optimize.minimize(_rosenbrock_numpy, initial[i], args=(shift,),
                                                jac=_rosenbrock_numpy_grad, method="BFGS", options={"gtol": 1e-8})
            self.assertTrue(np.allclose(tu.to_numpy(result.position[i]), reference.x, atol=1e-3))
            self.assertTrue(np.allclose(reference.x, [shift, shift**2], atol=1e-3))

    def test_max_iterations(self):
        """max_iterations stops the minimization before convergence."""
        initial = torch.tensor([[-1.2, 1.0], [2.0, 2.0]], dtype=torch.float64)
        result = optimizers.batched_minimize(_rosenbrock_problem(np.ones(2)), initial, max_iterations=2)
        self.assertEqual(result.num_iterations, 2)
        self.assertFalse(bool(result.converged.any()))
        self.assertFalse(bool(result.failed.any()))
        initial_value, _ = _rosenbrock_problem(np.ones(2))(initial)
        self.assertTrue(bool((result.objective_value < initial_value).all()))

    def test_initial_inverse_hessian_single(self):
        """The exact inverse Hessian ``(d, d)`` makes the first step a Newton step."""
        centres = self.rng.normal(size=(4, 3))
        hessian = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.0]])
        initial = torch.zeros(4, 3, dtype=tu.get_precision())
        result = optimizers.batched_minimize(
            _quadratic_problem(centres, hessian), initial,
            initial_inverse_hessian=torch.as_tensor(np.linalg.inv(hessian), dtype=tu.get_precision()),
            tolerance=_gradient_tolerance())
        self.assertTrue(bool(result.converged.all()))
        self.assertLessEqual(result.num_iterations, 2)
        self.assertTrue(np.allclose(tu.to_numpy(result.position), centres, atol=1e-4))

    def test_initial_inverse_hessian_batched(self):
        """A batch ``(B, d, d)`` of initial inverse Hessians is accepted, one per problem."""
        centres = self.rng.normal(size=(3, 2))
        hessians = np.array([np.diag([1.0, 10.0]), np.diag([5.0, 0.5]), np.array([[2.0, 0.9], [0.9, 1.0]])])
        initial = torch.zeros(3, 2, dtype=tu.get_precision())
        inverse_hessians = np.linalg.inv(hessians)
        result = optimizers.batched_minimize(_quadratic_problem(centres, hessians), initial,
                                             initial_inverse_hessian=inverse_hessians,
                                             tolerance=_gradient_tolerance())
        self.assertTrue(bool(result.converged.all()))
        self.assertLessEqual(result.num_iterations, 2)
        self.assertTrue(np.allclose(tu.to_numpy(result.position), centres, atol=1e-4))
        # a single shared inverse Hessian needs more iterations on these problems:
        shared = optimizers.batched_minimize(_quadratic_problem(centres, hessians), initial,
                                             initial_inverse_hessian=inverse_hessians[0],
                                             tolerance=_gradient_tolerance())
        self.assertGreater(shared.num_iterations, result.num_iterations)

    def test_non_finite_initial_value_fails(self):
        """A non-finite initial objective marks the problem as failed, the others converge."""
        base = _quadratic_problem(np.ones((3, 2)), np.eye(2))

        def value_and_grad(x):
            """Quadratics with a NaN objective on the second row."""
            value, grad = base(x)
            value = value.clone()
            value[1] = float("nan")
            return value, grad

        result = optimizers.batched_minimize(value_and_grad, torch.zeros(3, 2, dtype=tu.get_precision()),
                                             tolerance=_gradient_tolerance())
        self.assertEqual(result.failed.tolist(), [False, True, False])
        self.assertEqual(result.converged.tolist(), [True, False, True])
        self.assertTrue(np.allclose(tu.to_numpy(result.position[1]), 0.0))

    def test_non_finite_line_search_fails(self):
        """A line search that only finds non-finite values marks the problem as failed."""

        def value_and_grad(x):
            """Quadratic around 3 that is infinite away from the origin."""
            value = ((x - 3.0)**2).sum(-1)
            value = torch.where(x.abs().amax(-1) <= 1e-6, value, torch.full_like(value, float("inf")))
            return value, 2.0 * (x - 3.0)

        result = optimizers.batched_minimize(value_and_grad, torch.zeros(2, 2, dtype=tu.get_precision()),
                                             max_line_search_iterations=5)
        self.assertTrue(bool(result.failed.all()))
        self.assertFalse(bool(result.converged.any()))
        self.assertEqual(result.num_iterations, 1)
        self.assertTrue(np.allclose(tu.to_numpy(result.position), 0.0))

    def test_accepts_numpy_objective(self):
        """The objective can return numpy arrays and lists."""

        def value_and_grad(x):
            """Quadratic returning numpy arrays."""
            array = tu.to_numpy(x)
            return np.sum((array - 2.0)**2, axis=-1), 2.0 * (array - 2.0)

        result = optimizers.batched_minimize(value_and_grad, torch.zeros(2, 2, dtype=tu.get_precision()),
                                             tolerance=_gradient_tolerance())
        self.assertTrue(bool(result.converged.all()))
        self.assertTrue(np.allclose(tu.to_numpy(result.position), 2.0, atol=1e-4))

    def test_non_descent_initial_hessian_is_reset(self):
        """A negative definite initial inverse Hessian is reset to steepest descent and still converges."""
        centres = np.array([[1.0, -2.0], [0.5, 0.5]])
        hessian = np.array([[2.0, 0.3], [0.3, 1.0]])
        initial = torch.zeros(2, 2, dtype=tu.get_precision())
        tolerance = _gradient_tolerance()
        result = optimizers.batched_minimize(_quadratic_problem(centres, hessian), initial,
                                             initial_inverse_hessian=-np.eye(2), tolerance=tolerance,
                                             max_iterations=100)
        self.assertTrue(bool(result.converged.all()))
        self.assertFalse(bool(result.failed.any()))
        self.assertTrue(np.allclose(tu.to_numpy(result.position), centres, atol=10 * tolerance))
        # after the reset the BFGS estimates are positive definite:
        estimates = tu.to_numpy(result.inverse_hessian_estimate).astype(np.float64)
        self.assertTrue(np.all(np.linalg.eigvalsh(estimates) > 0.0))

#########################################################################################################
# Profiler helpers


class TestProfilerTorchHelpers(unittest.TestCase):
    """_insert_fixed_columns, _masked_gradient_ascent and the torch branch of points_minimizer."""

    def test_insert_fixed_columns_round_trip(self):
        """Removing and re-inserting fixed columns gives back the input."""
        torch.manual_seed(0)
        full = torch.randn(4, 5, dtype=tu.get_precision())
        for fixed_indices in ([0], [4], [1, 3], [3, 1], [0, 2, 4]):
            with self.subTest(fixed_indices=fixed_indices):
                free_indices = [i for i in range(5) if i not in fixed_indices]
                rebuilt = fp._insert_fixed_columns(full[:, free_indices], fixed_indices, full[:, fixed_indices])
                self.assertEqual(rebuilt.shape, full.shape)
                self.assertTrue(torch.equal(rebuilt, full))

    def test_insert_fixed_columns_gradient(self):
        """Gradients flow to the free coordinates only."""
        x_free = torch.zeros(2, 2, dtype=tu.get_precision(), requires_grad=True)
        fixed_values = torch.ones(2, 1, dtype=tu.get_precision())
        full = fp._insert_fixed_columns(x_free, [1], fixed_values)
        weights = torch.tensor([1.0, 2.0, 3.0], dtype=tu.get_precision())
        (full * weights).sum().backward()
        self.assertTrue(np.allclose(tu.to_numpy(x_free.grad), [[1.0, 3.0], [1.0, 3.0]]))

    def test_masked_gradient_ascent_moves_unmasked(self):
        """The ascent moves only unmasked coordinates while improving the log probability."""
        mean = np.array([0.5, -0.2, 1.0])
        precision_matrix = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 3.0]])
        flow = GaussianStubFlow(mean, precision_matrix)
        profiler = fp.posterior_profile_plotter(flow, initialize_cache=False, feedback=0)
        profiler.temp_inv_cov = precision_matrix
        ensemble = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0], [0.5, 0.3, 1.0]])
        initial_logP = tu.to_numpy(flow.log_probability(ensemble))
        mask = np.array([1.0, 0.0, 1.0])
        new_ensemble, logP, num_moving, num_iter = profiler._masked_gradient_ascent(
            learning_rate=0.1, num_iterations=200, ensemble=ensemble, mask=mask)
        self.assertIsInstance(new_ensemble, np.ndarray)
        self.assertIsInstance(logP, np.ndarray)
        self.assertIsInstance(num_moving, int)
        self.assertIsInstance(num_iter, int)
        self.assertEqual(new_ensemble.shape, ensemble.shape)
        self.assertTrue(np.array_equal(new_ensemble[:, 1], ensemble[:, 1].astype(new_ensemble.dtype)))
        self.assertTrue(np.all(logP >= initial_logP))
        self.assertTrue(np.all(logP[:2] > initial_logP[:2]))
        self.assertTrue(np.allclose(logP, tu.to_numpy(flow.log_probability(new_ensemble)), atol=1e-5))
        # the free coordinates approach the conditional maximum:
        free = [0, 2]
        block = precision_matrix[np.ix_(free, free)]
        conditional = mean[free] - np.linalg.solve(block, precision_matrix[np.ix_(free, [1])] @
                                                   (ensemble[:, 1] - mean[1])[None, :]).T
        start_distance = np.abs(ensemble[:, free] - conditional).max()
        end_distance = np.abs(new_ensemble[:, free] - conditional).max()
        self.assertLess(end_distance, 0.1 * start_distance)
        self.assertLessEqual(num_iter, 200)

    def test_points_minimizer_torch_branch(self):
        """The torch branch of points_minimizer minimizes all points and returns numpy arrays."""
        centres = np.array([[1.0, -1.0], [0.5, 2.0], [-3.0, 0.0]])
        value_and_grad = _quadratic_problem(centres, np.array([[2.0, 0.3], [0.3, 1.0]]))
        success, value, point = fp.points_minimizer(
            lambda x: value_and_grad(x)[0], lambda x: value_and_grad(x)[1], np.zeros((3, 2)),
            use_scipy=False, tolerance=_gradient_tolerance(), max_iterations=100, unrelated_option=True)
        self.assertIsInstance(success, np.ndarray)
        self.assertIsInstance(value, np.ndarray)
        self.assertIsInstance(point, np.ndarray)
        self.assertEqual(success.shape, (3,))
        self.assertEqual(value.shape, (3,))
        self.assertEqual(point.shape, (3, 2))
        self.assertTrue(success.all())
        self.assertTrue(np.allclose(point, centres, atol=1e-4))
        self.assertTrue(np.allclose(value, 0.0, atol=1e-6))

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
