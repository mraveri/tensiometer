"""Tests for the flow profiler, with lightweight stub flows and a real Gaussian flow."""

#########################################################################################################
# Imports

import contextlib
import io
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import scipy.optimize
import scipy.stats
import torch
from getdist import MCSamples
from getdist.densities import Density1D, Density2D

import tensiometer.synthetic_probability.flow_profiler as fp
import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import bijectors as bj
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Test configuration

GETDIST_SETTINGS = {"ignore_rows": 0.0, "smooth_scale_1D": 0.3, "smooth_scale_2D": 0.3}
GAUSSIAN_MEAN_3D = np.array([0.05, -0.03, 0.02])
GAUSSIAN_PRECISION_3D = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 4.0]])
FLOW_PROFILER = "tensiometer.synthetic_probability.flow_profiler."

#########################################################################################################
# Helpers


def _python_function(function):
    """Return the python implementation of a numba jitted function (or the function itself)."""
    return getattr(function, "py_func", function)


def _quiet():
    """Context manager that silences prints of the code under test."""
    return contextlib.redirect_stdout(io.StringIO())


def _fake_minimize_result(position, value, converged=True):
    """Build a fake :func:`batched_minimize` result with tensors in the active precision."""
    position = torch.as_tensor(np.atleast_2d(position), dtype=tu.get_precision())
    batch = position.shape[0]
    value = torch.as_tensor(np.broadcast_to(value, (batch,)).copy(), dtype=tu.get_precision())
    converged = torch.as_tensor(np.broadcast_to(converged, (batch,)).copy())
    return types.SimpleNamespace(converged=converged, failed=~converged, num_iterations=1,
                                 objective_value=value, objective_gradient=torch.zeros_like(position),
                                 position=position, inverse_hessian_estimate=None)

#########################################################################################################
# Stub flows


class StubFlow:
    """Flow stub with a flat log probability and seeded Gaussian samples."""

    def __init__(self, num_params=2, with_ranges=True, lower=-1.0, upper=1.0, sample_scale=0.3, seed=0):
        """Build the stub.

        :param num_params: number of parameters
        :param with_ranges: attach parameter ranges ``(lower, upper)`` to all parameters
        :param lower: lower range bound
        :param upper: upper range bound
        :param sample_scale: standard deviation of the samples around the range centre
        :param seed: seed of the sample generator
        """
        self.num_params = num_params
        self.param_names = ["p" + str(i) for i in range(num_params)]
        self.param_labels = ["P" + str(i) for i in range(num_params)]
        if with_ranges:
            self.parameter_ranges = {name: (lower, upper) for name in self.param_names}
        else:
            self.parameter_ranges = None
        self.name_tag = "stub"
        self.center = 0.5 * (lower + upper)
        self.sample_scale = sample_scale
        self.generator = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.chain_samples = (self.center + 0.1 * rng.standard_normal((50, num_params))).astype(tu.np_prec)
        self.chain_loglikes = np.zeros(50, dtype=tu.np_prec)

    def cast(self, arr):
        """Convert to a CPU tensor in the active precision."""
        return tu.to_tensor(arr, device="cpu")

    def sample(self, n):
        """Draw seeded Gaussian samples."""
        noise = torch.randn(n, self.num_params, generator=self.generator, dtype=tu.get_precision())
        return self.center + self.sample_scale * noise

    def log_probability(self, x):
        """Flat log probability."""
        x = tu.to_tensor(x)
        return torch.zeros(x.shape[0], dtype=tu.get_precision())

    def log_probability_abs(self, x):
        """Flat log probability in abstract coordinates."""
        return self.log_probability(x)

    def log_probability_jacobian(self, x):
        """Zero gradient."""
        return torch.zeros_like(tu.to_tensor(x))

    def log_probability_abs_jacobian(self, x):
        """Zero gradient in abstract coordinates."""
        return self.log_probability_jacobian(x)

    def map_to_abstract_coord(self, x):
        """Identity map."""
        return tu.to_tensor(x)

    def map_to_original_coord(self, x):
        """Identity map."""
        return tu.to_tensor(x)


class ShiftedStubFlow(StubFlow):
    """Stub whose abstract coordinates are the parameters shifted by ``-10``."""

    def map_to_abstract_coord(self, x):
        """Shift by -10."""
        return tu.to_tensor(x) - 10.0

    def map_to_original_coord(self, x):
        """Shift by +10."""
        return tu.to_tensor(x) + 10.0


class GaussianStubFlow(StubFlow):
    """Stub with a Gaussian log probability (differentiable through torch)."""

    def __init__(self, mean, precision_matrix, **kwargs):
        """Build the stub.

        :param mean: mean, shape ``(D,)``
        :param precision_matrix: inverse covariance, shape ``(D, D)``
        :param kwargs: passed to :class:`StubFlow`
        """
        super().__init__(num_params=len(mean), **kwargs)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.precision_matrix = np.asarray(precision_matrix, dtype=np.float64)

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


class OffsetGaussianStubFlow(GaussianStubFlow):
    """3D Gaussian stub whose log probability is shifted by a constant."""

    def __init__(self, offset):
        """Build the stub.

        :param offset: constant added to the log probability
        """
        super().__init__(GAUSSIAN_MEAN_3D, GAUSSIAN_PRECISION_3D)
        self.offset = offset

    def log_probability(self, x):
        """Shifted Gaussian log probability."""
        return super().log_probability(x) + self.offset


class NonFiniteGaussianStubFlow(GaussianStubFlow):
    """1D Gaussian stub whose log probability is not finite above a threshold."""

    def __init__(self, threshold):
        """Build the stub.

        :param threshold: the log probability is NaN for parameter values above it
        """
        super().__init__([0.0], np.array([[4.0]]))
        self.threshold = threshold

    def log_probability(self, x):
        """Gaussian log probability, NaN above the threshold."""
        log_P = super().log_probability(x)
        x = tu.to_tensor(x)
        return torch.where(x[:, 0] > self.threshold, torch.full_like(log_P, float("nan")), log_P)


class StubProfiler(fp.posterior_profile_plotter):
    """Profiler with a fixed profile population and stubbed marginal densities."""

    population = np.array([[0.2, 0.3], [0.6, 0.7]])
    population_logP = np.array([0.1, 0.2])

    def __init__(self, flow):
        """Build the profiler without caches or polishing."""
        super().__init__(flow, initialize_cache=False, feedback=0, use_scipy=False,
                         pre_polish=False, polish=False, smoothing=False)

    def sample_profile_population(self, **kwargs):
        """Set a fixed population."""
        self.temp_samples = np.array(self.population, dtype=tu.np_prec)
        self.temp_probs = np.array(self.population_logP, dtype=tu.np_prec)
        self.temp_cov = np.eye(self.n)
        self.temp_inv_cov = np.eye(self.n)
        return None

    def get1DDensityGridData(self, name_or_idx, **kwargs):
        """Stub 1D density."""
        return types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]))

    def get2DDensityGridData(self, ind1, ind2, **kwargs):
        """Stub 2D density."""
        return types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]), y=np.array([0.0, 0.5, 1.0]))


def _profiler(flow=None, **kwargs):
    """Build a profiler on a stub flow without computing caches."""
    if flow is None:
        flow = StubFlow()
    return fp.posterior_profile_plotter(flow, initialize_cache=False, feedback=0, **kwargs)


def _gaussian_stub_3D(**kwargs):
    """Correlated 3D Gaussian stub with :data:`GAUSSIAN_MEAN_3D` and :data:`GAUSSIAN_PRECISION_3D`."""
    return GaussianStubFlow(GAUSSIAN_MEAN_3D, GAUSSIAN_PRECISION_3D, **kwargs)


def _conditional_maximum(points, free_index):
    """Maximum of the 3D Gaussian stub along one coordinate, at the other coordinates of the points.

    :param points: points, shape ``(N, 3)``
    :param free_index: index of the maximized coordinate
    :returns: the maximizing values of the free coordinate, shape ``(N,)``
    """
    others = [i for i in range(3) if i != free_index]
    row = GAUSSIAN_PRECISION_3D[free_index]
    diff = points[:, others] - GAUSSIAN_MEAN_3D[others]
    return GAUSSIAN_MEAN_3D[free_index] - diff @ row[others] / row[free_index]


def _is_subset_of(rows, population):
    """Whether every row of ``rows`` is exactly one of the rows of ``population``."""
    distances = np.linalg.norm(rows[:, None, :] - population[None, :, :], axis=-1)
    return bool(np.all(np.min(distances, axis=1) == 0.0))

#########################################################################################################
# Module level helpers


class TestModuleHelpers(unittest.TestCase):
    """Binned argmax, points_minimizer and find_flow_MAP."""

    def test_binned_argmax_1D(self):
        """The index of the maximum of every bin is returned, -1 for empty bins."""
        bins = np.array([0, 1, 0, 1])
        vals = np.array([0.1, 0.2, 0.3, 0.4])
        result = _python_function(fp._binned_argmax_1D)(bins, vals, 3)
        self.assertEqual(list(result), [2, 3, -1])

    def test_binned_argmax_2D(self):
        """The index of the maximum of every 2D bin is returned, -1 for empty bins."""
        x_bins = np.array([0, 1, 1])
        y_bins = np.array([1, 0, 0])
        vals = np.array([0.5, 0.6, 0.7])
        result = _python_function(fp._binned_argmax_2D)(x_bins, y_bins, vals, 2, 2)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(result[1, 0], 2)
        self.assertEqual(result[0, 0], -1)
        self.assertEqual(result[1, 1], -1)

    def test_insert_fixed_columns(self):
        """Fixed columns are inserted at their indices."""
        x_free = torch.tensor([[1.0, 2.0]], dtype=tu.get_precision())
        fixed_values = torch.tensor([[10.0, 20.0]], dtype=torch.float64)
        full = fp._insert_fixed_columns(x_free, [2, 0], fixed_values)
        self.assertEqual(full.dtype, tu.get_precision())
        self.assertTrue(np.allclose(tu.to_numpy(full), [[20.0, 1.0, 10.0, 2.0]]))

    def test_points_minimizer_torch_branch(self):
        """The torch branch calls batched_minimize on a tensor and returns numpy arrays."""
        fake_result = _fake_minimize_result(np.zeros((2, 2)), 0.5)
        with patch("tensiometer.synthetic_probability.flow_profiler.batched_minimize", autospec=True,
                   return_value=fake_result) as mock_min:
            success, value, point = fp.points_minimizer(
                lambda x: (x**2).sum(-1), lambda x: 2 * x, np.array([[0.5, 0.5], [1.0, 1.0]]),
                bounds=None, use_scipy=False, tolerance=1e-3, not_an_option=1)
        args, kwargs = mock_min.call_args
        self.assertTrue(torch.is_tensor(args[1]))
        self.assertEqual(args[1].dtype, tu.get_precision())
        self.assertEqual(kwargs, {"tolerance": 1e-3})
        value_and_grad = args[0](args[1])
        self.assertTrue(np.allclose(tu.to_numpy(value_and_grad[0]), [0.5, 2.0]))
        for array in (success, value, point):
            self.assertIsInstance(array, np.ndarray)
        self.assertTrue(success.all())
        self.assertEqual(point.shape, (2, 2))
        self.assertTrue(np.allclose(value, 0.5))

    def test_points_minimizer_torch_branch_ignores_bounds(self):
        """The torch branch warns that bounds are ignored."""
        fake_result = _fake_minimize_result(np.zeros((1, 2)), 0.0)
        buffer = io.StringIO()
        with patch("tensiometer.synthetic_probability.flow_profiler.batched_minimize", return_value=fake_result), \
                contextlib.redirect_stdout(buffer):
            fp.points_minimizer(lambda x: x.sum(-1), lambda x: x, np.zeros((1, 2)),
                                bounds=[(-1, 1), (-1, 1)], use_scipy=False)
        self.assertIn("does not support bounds", buffer.getvalue())

    def test_points_minimizer_scipy_branch_jac(self):
        """The scipy branch passes the jacobian only when use_jac is True."""
        fake_res = types.SimpleNamespace(success=True, fun=1.0, x=np.array([1.0, -1.0]), nfev=1, njev=1,
                                         message="ok")

        def func(x):
            """Quadratic objective."""
            return np.sum(x**2)

        def jac(x):
            """Gradient of the quadratic objective."""
            return 2 * x

        for use_jac in (True, False):
            with self.subTest(use_jac=use_jac):
                with patch("tensiometer.synthetic_probability.flow_profiler.minimize",
                           return_value=fake_res) as mock_min, _quiet():
                    success, value, point = fp.points_minimizer(
                        func, jac, [np.array([0.5, -0.5])], bounds=[(-1, 1), (-1, 1)], use_scipy=True,
                        feedback=3, use_jac=use_jac, method="L-BFGS-B", options={"maxls": 10})
                kwargs = mock_min.call_args.kwargs
                if use_jac:
                    self.assertIs(kwargs["jac"], jac)
                else:
                    self.assertIsNone(kwargs["jac"])
                self.assertEqual(kwargs["method"], "L-BFGS-B")
                self.assertEqual(kwargs["options"], {"maxls": 10})
                self.assertEqual(kwargs["bounds"], [(-1, 1), (-1, 1)])
                self.assertTrue(success[0])
                self.assertEqual(value[0], 1.0)
                self.assertTrue(np.allclose(point, [[1.0, -1.0]]))

    def test_points_minimizer_scipy_real(self):
        """The scipy branch minimizes every point."""
        success, value, point = fp.points_minimizer(
            lambda x: np.sum((x - 1.0)**2), lambda x: 2 * (x - 1.0), np.array([[0.0, 0.0], [3.0, -2.0]]))
        self.assertTrue(success.all())
        self.assertTrue(np.allclose(point, 1.0, atol=1e-4))
        self.assertTrue(np.allclose(value, 0.0, atol=1e-6))

    def test_find_flow_MAP_abstract_and_box_raises(self):
        """Abstract coordinates and a box bijector cannot be used together."""
        with self.assertRaises(ValueError), _quiet():
            fp.find_flow_MAP(StubFlow(), abstract=True, box_bijector=bj.Identity(), initial_points=np.zeros((1, 2)))

    def test_find_flow_MAP_random_search(self):
        """Without initial points the best flow samples are followed."""
        fake_res = (np.array([True, True, True]), np.array([0.0, -1.0, 2.0]), np.arange(6.0).reshape(3, 2))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   return_value=fake_res) as mock_pm, _quiet():
            value, solution = fp.find_flow_MAP(StubFlow(), feedback=1, abstract=False, num_samples=20,
                                               num_best_to_follow=3)
        self.assertEqual(mock_pm.call_args.args[2].shape, (3, 2))
        self.assertEqual(mock_pm.call_args.kwargs["bounds"], [(-1.0, 1.0), (-1.0, 1.0)])
        self.assertEqual(value, 1.0)
        self.assertTrue(np.allclose(solution, [2.0, 3.0]))

    def test_find_flow_MAP_abstract_maps_back(self):
        """With abstract coordinates the initial points are mapped in and the solution out."""
        fake_res = (np.array([True]), np.array([0.0]), np.array([[0.1, 0.2]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   return_value=fake_res) as mock_pm:
            value, solution = fp.find_flow_MAP(ShiftedStubFlow(), feedback=0, abstract=True,
                                               initial_points=np.ones((1, 2)))
        self.assertTrue(np.allclose(mock_pm.call_args.args[2], [[-9.0, -9.0]]))
        self.assertIsNone(mock_pm.call_args.kwargs["bounds"])
        self.assertTrue(np.allclose(solution, [10.1, 10.2], atol=1e-5))

    def test_find_flow_MAP_box_bijector_maps_back(self):
        """With a box bijector the initial points are mapped in and the solution out."""
        fake_res = (np.array([True]), np.array([0.0]), np.array([[0.1, 0.2]]))
        box_bijector = bj.Shift(5.0)
        for use_scipy in (True, False):
            with self.subTest(use_scipy=use_scipy):
                with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                           return_value=fake_res) as mock_pm:
                    value, solution = fp.find_flow_MAP(StubFlow(), feedback=0, abstract=False,
                                                       box_bijector=box_bijector, initial_points=np.ones((1, 2)),
                                                       use_scipy=use_scipy)
                self.assertTrue(np.allclose(mock_pm.call_args.args[2], [[-4.0, -4.0]]))
                self.assertIsNone(mock_pm.call_args.kwargs["bounds"])
                self.assertTrue(np.allclose(solution, [5.1, 5.2], atol=1e-5))

    def test_find_flow_MAP_scipy_objective_uses_box_bijector(self):
        """The scipy objective and jacobian evaluate the flow through the box bijector."""
        flow = GaussianStubFlow([5.0, 5.0], np.eye(2))
        box_bijector = bj.Shift(5.0)
        recorded = {}

        def fake_points_minimizer(func, jac, x0, **kwargs):
            """Record the objective at the origin of the unbounded coordinates."""
            recorded["func"] = func(np.zeros(2))
            recorded["jac"] = jac(np.array([1.0, 0.0]))
            recorded["types"] = (type(recorded["func"]), type(recorded["jac"]))
            return np.array([True]), np.array([0.0]), np.zeros((1, 2))

        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   side_effect=fake_points_minimizer):
            fp.find_flow_MAP(flow, feedback=0, abstract=False, box_bijector=box_bijector,
                             initial_points=np.full((1, 2), 5.0), use_scipy=True)
        self.assertAlmostEqual(float(recorded["func"]), 0.0, places=6)
        self.assertTrue(np.allclose(recorded["jac"], [1.0, 0.0], atol=1e-6))
        self.assertEqual(recorded["jac"].dtype, np.float64)

    def test_find_flow_MAP_falls_back_to_best_finite(self):
        """When nothing converged the best finite point is used."""
        fake_res = (np.array([False, False, False]), np.array([1.0, 0.5, np.nan]),
                    np.array([[0.0, 0.0], [0.3, 0.4], [1.0, 1.0]]))
        buffer = io.StringIO()
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer", return_value=fake_res), \
                contextlib.redirect_stdout(buffer):
            value, solution = fp.find_flow_MAP(StubFlow(), feedback=1, abstract=False,
                                               initial_points=np.zeros((3, 2)))
        self.assertIn("no minimization converged", buffer.getvalue())
        self.assertEqual(value, -0.5)
        self.assertTrue(np.allclose(solution, [0.3, 0.4]))

    def test_find_flow_MAP_prefers_converged(self):
        """Converged points are preferred over better non-converged ones."""
        fake_res = (np.array([False, True]), np.array([-5.0, 0.5]), np.array([[0.0, 0.0], [0.3, 0.4]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer", return_value=fake_res):
            value, solution = fp.find_flow_MAP(StubFlow(), feedback=0, abstract=False,
                                               initial_points=np.zeros((2, 2)))
        self.assertEqual(value, -0.5)
        self.assertTrue(np.allclose(solution, [0.3, 0.4]))

    def test_find_flow_MAP_no_finite_raises(self):
        """A ValueError is raised when no minimization gave a finite result."""
        fake_res = (np.array([True]), np.array([np.inf]), np.array([[np.nan, 0.0]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer", return_value=fake_res):
            with self.assertRaises(ValueError):
                fp.find_flow_MAP(StubFlow(), feedback=0, abstract=False, initial_points=np.zeros((1, 2)))

#########################################################################################################
# Options and caches


class TestProfilerOptionsAndCaches(unittest.TestCase):
    """Options, cache handling and the profile population."""

    def test_options_are_per_instance(self):
        """Keyword arguments update the instance options only."""
        profiler = _profiler(num_points_1D=8)
        self.assertEqual(profiler.options["num_points_1D"], 8)
        profiler.options["scipy_options"]["gtol"] = 1.0
        other = _profiler()
        self.assertEqual(other.options["num_points_1D"], 64)
        self.assertEqual(other.options["scipy_options"], {"ftol": 1.e-6, "gtol": 1.e-05})
        self.assertEqual(fp.posterior_profile_plotter.options["num_points_1D"], 64)
        self.assertEqual(fp.posterior_profile_plotter.options["scipy_options"], {"ftol": 1.e-6, "gtol": 1.e-05})

    def test_torch_option_names(self):
        """The minimizer options use the torch_ prefix."""
        profiler = _profiler()
        for key in ("torch_tolerance", "torch_max_iterations", "torch_max_line_search_iterations"):
            self.assertIn(key, profiler.options)
        self.assertFalse(any(key.startswith("tf_") for key in profiler.options))

    def test_minimizer_options(self):
        """_minimizer_options reads kwargs first and the profiler options otherwise."""
        profiler = _profiler(torch_max_iterations=7)
        options = profiler._minimizer_options({"torch_tolerance": 1e-3})
        self.assertEqual(options, {"tolerance": 1e-3, "max_iterations": 7,
                                   "max_line_search_iterations": profiler.options["torch_max_line_search_iterations"]})

    def test_extra_options_are_stored(self):
        """Unknown keyword arguments are stored in the options and do not reach getdist."""
        profiler = _profiler(permutations=[np.array([0, 1])], map_to_unitcube=True, transformation_type="spline")
        self.assertTrue(profiler.options["map_to_unitcube"])

    def test_feedback_prints(self):
        """Feedback prints the settings."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fp.posterior_profile_plotter(StubFlow(), feedback=1)
        self.assertIn("use_scipy", buffer.getvalue())

    def test_init_without_chain_samples(self):
        """Without chain samples the margestats come from flow samples."""
        flow = StubFlow()
        flow.chain_samples = None
        profiler = _profiler(flow)
        self.assertEqual(profiler.samples.shape, (1000 * flow.num_params, flow.num_params))

    def test_par_and_number_missing(self):
        """_parAndNumber returns None for unknown parameters."""
        idx, name = _profiler()._parAndNumber("missing")
        self.assertIsNone(idx)
        self.assertIsNone(name)

    def test_reset_cache(self):
        """reset_cache clears the caches."""
        profiler = _profiler()
        profiler.temp_samples = np.ones((1, 2))
        profiler.temp_probs = np.ones(1)
        profiler.flow_MAP = np.zeros(2)
        profiler.profile_density_1D = {"x": 1}
        profiler.profile_density_2D = {"x": 1}
        profiler.reset_cache()
        self.assertIsNone(profiler.temp_samples)
        self.assertIsNone(profiler.temp_probs)
        self.assertIsNone(profiler.flow_MAP)
        self.assertEqual(profiler.profile_density_1D, {})
        self.assertEqual(profiler.profile_density_2D, {})

    def test_update_cache_no_updates(self):
        """update_cache without updates only cleans up."""
        profiler = _profiler()
        profiler.update_cache(update_MAP=False, update_1D=False, update_2D=False)
        self.assertIsNone(profiler.temp_samples)
        self.assertIsNone(profiler.temp_probs)

    def test_update_cache_iterative_no_updates(self):
        """update_cache_iterative without updates only cleans up."""
        profiler = _profiler()
        profiler.update_cache_iterative(update_1D=False, update_2D=False, niter=0)
        self.assertIsNone(profiler.temp_samples)
        self.assertIsNone(profiler.temp_probs)

    def test_sample_profile_population_numpy_caches(self):
        """The profile population caches are numpy arrays."""
        profiler = _profiler()
        profiler.sample_profile_population(num_minimization_samples=500)
        for cache in (profiler.temp_samples, profiler.temp_probs, profiler.temp_cov, profiler.temp_inv_cov):
            self.assertIsInstance(cache, np.ndarray)
        self.assertEqual(profiler.temp_samples.shape, (500, 2))
        self.assertEqual(profiler.temp_probs.shape, (500,))
        self.assertTrue(np.allclose(profiler.temp_cov, np.cov(profiler.temp_samples.T)))
        self.assertTrue(np.allclose(profiler.temp_cov @ profiler.temp_inv_cov, np.eye(2), atol=1e-4))

    def test_sample_profile_population_box_prior(self):
        """With a box prior the covariance is computed in the unbounded coordinates."""
        flow = StubFlow(lower=0.0, upper=1.0, sample_scale=0.1)
        profiler = _profiler(flow, box_prior=True)
        profiler.feedback = 2
        with _quiet():
            profiler.sample_profile_population(num_minimization_samples=500)
        self.assertEqual(profiler.temp_samples.shape, (500, 2))
        self.assertEqual(profiler.temp_probs.shape, (500,))
        with torch.no_grad():
            unbounded = tu.to_numpy(profiler._get_masked_box_bijector().inverse(profiler.temp_samples))
        self.assertTrue(np.allclose(profiler.temp_cov, np.cov(unbounded.T), rtol=1e-3, atol=1e-5))
        self.assertGreater(profiler.temp_cov[0, 0], np.cov(profiler.temp_samples.T)[0, 0])
        self.assertIsInstance(profiler.temp_inv_cov, np.ndarray)

    def test_sample_profile_population_box_prior_discards_outside(self):
        """With a box prior samples outside the box are discarded."""
        flow = StubFlow(lower=0.0, upper=1.0, sample_scale=0.5)
        profiler = _profiler(flow, box_prior=True)
        profiler.sample_profile_population(num_minimization_samples=500)
        self.assertLess(profiler.temp_samples.shape[0], 500)
        self.assertTrue(np.all((profiler.temp_samples > 0.0) & (profiler.temp_samples < 1.0)))

    def test_masked_box_bijector(self):
        """The box bijector maps unbounded coordinates to the ranges of the unmasked parameters."""
        flow = StubFlow(lower=0.0, upper=2.0)
        profiler = _profiler(flow)
        bijector = profiler._get_masked_box_bijector()
        self.assertIsInstance(bijector, bj.Bijector)
        with torch.no_grad():
            centre = tu.to_numpy(bijector(torch.zeros(1, 2, dtype=tu.get_precision())))
            masked = tu.to_numpy(profiler._get_masked_box_bijector(mask=[1, 0])(
                torch.tensor([[-50.0], [0.0], [50.0]], dtype=tu.get_precision())))
        self.assertTrue(np.allclose(centre, [[1.0, 1.0]]))
        self.assertEqual(masked.shape, (3, 1))
        self.assertTrue(np.allclose(masked[:, 0], [0.0, 1.0, 2.0], atol=1e-5))
        no_ranges = _profiler(StubFlow(with_ranges=False))
        self.assertIsInstance(no_ranges._get_masked_box_bijector(), bj.Identity)

#########################################################################################################
# MAP finding


class TestProfilerMAP(unittest.TestCase):
    """find_MAP branches with stub flows."""

    def test_find_MAP_box_prior_maps_back(self):
        """With a box prior the MAP found in unbounded coordinates is mapped to the box."""
        flow = StubFlow(lower=0.0, upper=2.0)
        profiler = _profiler(flow, box_prior=True, use_scipy=False)
        profiler.sample_profile_population(num_minimization_samples=50)
        fake_res = (np.array([True, True]), np.array([0.0, -1.0]), np.zeros((2, 2)))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   return_value=fake_res) as mock_pm:
            value, solution = profiler.find_MAP(num_best_to_follow=2)
        self.assertEqual(mock_pm.call_args.args[2].shape, (2, 2))
        self.assertEqual(value, 1.0)
        self.assertTrue(np.allclose(solution, [1.0, 1.0], atol=1e-5))
        self.assertTrue(np.allclose(profiler.flow_MAP, solution))
        self.assertEqual(profiler.flow_MAP_logP, value)
        self.assertIsNotNone(profiler.bestfit)

    def test_find_MAP_scipy_options(self):
        """The scipy branch forwards the scipy options."""
        profiler = _profiler(use_scipy=True, scipy_method="BFGS", scipy_use_jac=False)
        profiler.temp_samples = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=tu.np_prec)
        profiler.temp_probs = np.array([0.1, 0.2], dtype=tu.np_prec)
        fake_res = (np.array([True]), np.array([0.0]), np.array([[0.1, 0.2]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   return_value=fake_res) as mock_pm:
            value, solution = profiler.find_MAP(randomize=True, num_best_to_follow=1)
        kwargs = mock_pm.call_args.kwargs
        self.assertEqual(kwargs["method"], "BFGS")
        self.assertFalse(kwargs["use_jac"])
        self.assertEqual(kwargs["options"], profiler.options["scipy_options"])
        self.assertTrue(np.allclose(mock_pm.call_args.args[2], [[0.5, 0.5]]))
        self.assertTrue(np.allclose(solution, [0.1, 0.2]))

    def test_find_MAP_torch_options(self):
        """The torch branch forwards the torch options."""
        profiler = _profiler(use_scipy=False, torch_max_iterations=3)
        fake_res = (np.array([True]), np.array([0.0]), np.array([[0.1, 0.2]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   return_value=fake_res) as mock_pm:
            profiler.find_MAP(x0=np.zeros((1, 2)), torch_tolerance=1e-2)
        kwargs = mock_pm.call_args.kwargs
        self.assertEqual(kwargs["tolerance"], 1e-2)
        self.assertEqual(kwargs["max_iterations"], 3)
        self.assertFalse(kwargs["use_scipy"])

    def test_find_MAP_x0_real_minimizer(self):
        """find_MAP from x0 with the real batched minimizer on a Gaussian stub."""
        flow = GaussianStubFlow([0.2, -0.1], np.array([[2.0, 0.3], [0.3, 1.0]]), with_ranges=False)
        profiler = _profiler(flow, use_scipy=False)
        value, solution = profiler.find_MAP(x0=np.zeros((2, 2)), abstract=True)
        self.assertEqual(solution.shape, (2,))
        self.assertTrue(np.allclose(solution, [0.2, -0.1], atol=1e-3))
        self.assertAlmostEqual(value, 0.0, places=4)

    def test_find_MAP_without_randomization(self):
        """Without randomization the best chain samples are followed."""
        flow = StubFlow()
        flow.chain_loglikes = np.arange(50, dtype=tu.np_prec)
        profiler = _profiler(flow, use_scipy=True)
        fake_res = (np.array([True]), np.array([0.0]), np.array([[0.1, 0.2]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   return_value=fake_res) as mock_pm:
            profiler.find_MAP(randomize=False, num_best_to_follow=3)
        initial = mock_pm.call_args.args[2]
        self.assertEqual(initial.shape, (3, 2))
        expected = np.sort(flow.chain_samples[:3], axis=0)
        self.assertTrue(np.allclose(np.sort(initial, axis=0), expected, atol=1e-6))

#########################################################################################################
# Likelihood statistics and best fit


class TestProfilerLikeStatsBestFit(unittest.TestCase):
    """Best fit and likelihood statistics."""

    def _profiler_with_MAP(self):
        """Profiler with a cached MAP and population."""
        profiler = _profiler()
        profiler.flow_MAP_logP = 0.0
        profiler.flow_MAP = np.array([0.1, -0.2])
        profiler.temp_samples = np.array([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0]], dtype=tu.np_prec)
        profiler.temp_probs = np.array([-0.1, -1.0, -3.0], dtype=tu.np_prec)
        return profiler

    def test_errors_without_MAP(self):
        """Best fit and likestats need the MAP."""
        profiler = _profiler()
        with self.assertRaises(ValueError):
            profiler.getBestFit()
        with self.assertRaises(ValueError):
            profiler._initialize_likestats()
        with self.assertRaises(ValueError):
            profiler._initialize_bestfit()

    def test_bestfit(self):
        """getBestFit returns the cached MAP."""
        profiler = self._profiler_with_MAP()
        profiler._initialize_bestfit()
        bestfit = profiler.getBestFit()
        self.assertEqual(len(bestfit.names), profiler.n)
        self.assertEqual([par.best_fit for par in bestfit.names], [0.1, -0.2])
        self.assertEqual([par.name for par in bestfit.names], ["p0", "p1"])
        self.assertEqual(profiler.logLike, 0.0)

    def _getdist_likestats(self, profiler):
        """getdist likelihood statistics of the population, with loglikes minus the flow log probability."""
        with _quiet():
            chain = MCSamples(samples=np.asarray(profiler.temp_samples, dtype=np.float64),
                              loglikes=-np.asarray(profiler.temp_probs, dtype=np.float64),
                              names=["p0", "p1"], settings=GETDIST_SETTINGS)
            return chain.getLikeStats()

    def test_likestats_without_profile_limits(self):
        """Likelihood statistics follow the getdist convention of loglikes as minus the log probability."""
        profiler = self._profiler_with_MAP()
        stats = profiler.getLikeStats(profile_lims=False)
        reference = self._getdist_likestats(profiler)
        self.assertIs(profiler.likeStats, stats)
        self.assertTrue(hasattr(stats, "ND_contours"))
        for key in ["meanLogLike", "logMeanLike", "logMeanInvLike", "varLogLike"]:
            self.assertAlmostEqual(getattr(stats, key), getattr(reference, key), places=5, msg=key)
        self.assertAlmostEqual(stats.meanLogLike, -np.mean(profiler.temp_probs), places=5)
        self.assertEqual(stats.logLike_sample, -profiler.flow_MAP_logP)
        self.assertAlmostEqual(stats.complexity, 2.0 * (stats.meanLogLike - stats.logLike_sample))
        for j, par in enumerate(profiler.paramNames.names):
            self.assertEqual(par.bestfit_sample, profiler.flow_MAP[j])

    def test_likestats_without_profile_limits_region_bounds(self):
        """The N-dimensional limits are the extremes of the highest probability samples, as in getdist."""
        profiler = self._profiler_with_MAP()
        stats = profiler.getLikeStats(profile_lims=False)
        reference = self._getdist_likestats(profiler)
        indexes = np.argsort(-profiler.temp_probs)
        for j, par in enumerate(profiler.paramNames.names):
            self.assertTrue(np.allclose(par.ND_limit_bot, reference.names[j].ND_limit_bot))
            self.assertTrue(np.allclose(par.ND_limit_top, reference.names[j].ND_limit_top))
            for i, cont in enumerate(stats.ND_contours):
                region = profiler.temp_samples[indexes[:cont], j]
                self.assertEqual(par.ND_limit_bot[i], np.min(region))
                self.assertEqual(par.ND_limit_top[i], np.max(region))

    def _gaussian_profile(self, name, **kwargs):
        """Unit Gaussian profile on a fine grid."""
        x = np.linspace(-4.0, 4.0, 1601)
        return Density1D(x, np.exp(-0.5 * x**2))

    def test_likestats_with_profile_limits(self):
        """Profile limits are symmetric around the peak of a Gaussian profile."""
        profiler = self._profiler_with_MAP()
        with patch.object(fp.posterior_profile_plotter, "get1DDensityGridData", side_effect=self._gaussian_profile):
            stats = profiler.getLikeStats(profile_lims=True)
        self.assertIsNotNone(stats)
        for j, par in enumerate(profiler.paramNames.names):
            self.assertEqual(len(par.ND_limit_bot), len(profiler.contours))
            self.assertTrue(np.all(par.ND_limit_bot < 0.0))
            self.assertTrue(np.all(par.ND_limit_top > 0.0))
            self.assertTrue(np.allclose(par.ND_limit_bot, -par.ND_limit_top, atol=1e-2))
            self.assertTrue(np.all(np.diff(par.ND_limit_top) > 0.0))
            self.assertEqual(par.bestfit_sample, profiler.flow_MAP[j])

    def test_likestats_profile_limits_gaussian_values(self):
        """For a unit Gaussian profile the limits are at +-sqrt(chi2.ppf(cont, 1)) (2 Delta log P = chi2)."""
        profiler = self._profiler_with_MAP()
        with patch.object(fp.posterior_profile_plotter, "get1DDensityGridData", side_effect=self._gaussian_profile):
            profiler.getLikeStats(profile_lims=True)
        expected = np.sqrt(scipy.stats.chi2.ppf(profiler.contours, 1))
        for par in profiler.paramNames.names:
            self.assertTrue(np.allclose(par.ND_limit_top, expected, atol=2e-2))
            self.assertTrue(np.allclose(par.ND_limit_bot, -expected, atol=2e-2))

    def test_likestats_profile_limits_one_sided(self):
        """When a profile crosses the contours on one side only, the other limit is the parameter range bound."""
        for lower, upper in ((-4.0, 0.0), (0.0, 4.0)):
            with self.subTest(lower=lower, upper=upper):
                profiler = self._profiler_with_MAP()
                x = np.linspace(lower, upper, 801)
                half_gaussian = Density1D(x, np.exp(-0.5 * x**2))
                with patch.object(fp.posterior_profile_plotter, "get1DDensityGridData", return_value=half_gaussian):
                    profiler.getLikeStats(profile_lims=True)
                expected = np.sqrt(scipy.stats.chi2.ppf(profiler.contours, 1))
                for par in profiler.paramNames.names:
                    if lower < 0.0:
                        self.assertTrue(np.allclose(par.ND_limit_bot, -expected, atol=2e-2))
                        self.assertTrue(np.allclose(par.ND_limit_top, par.limmax))
                    else:
                        self.assertTrue(np.allclose(par.ND_limit_bot, par.limmin))
                        self.assertTrue(np.allclose(par.ND_limit_top, expected, atol=2e-2))

    def test_likestats_samples_population_when_missing(self):
        """The population is sampled when it is not cached."""
        profiler = _profiler()
        profiler.flow_MAP_logP = 0.0
        profiler.flow_MAP = np.zeros(2)
        profiler.options["num_minimization_samples"] = 40
        profiler.getLikeStats(profile_lims=False)
        self.assertEqual(profiler.temp_samples.shape, (40, 2))

#########################################################################################################
# Profiles with stubs


class TestProfilerProfilesStub(unittest.TestCase):
    """Profile helpers, polishing and gradient ascent with stub flows."""

    def test_get1DDensityGridData_cached(self):
        """Cached 1D profiles are returned."""
        profiler = _profiler()
        cached_density = Density1D(np.linspace(-1, 1, 3), np.ones(3))
        profiler.profile_density_1D = {0: cached_density}
        self.assertIs(profiler.get1DDensityGridData("p0", num_points_1D=4), cached_density)

    def test_get2DDensityGridData_cached(self):
        """Cached 2D profiles are returned."""
        profiler = _profiler()
        cached_density = Density2D(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3), np.ones((3, 3)))
        profiler.profile_density_2D = {0: {1: cached_density}}
        self.assertIs(profiler.get2DDensityGridData("p0", "p1"), cached_density)

    def test_precompute_calls_getters(self):
        """precompute_1D and precompute_2D call the profile getters."""
        profiler = _profiler()
        with patch.object(fp.posterior_profile_plotter, "get1DDensityGridData") as mock_get1d, \
                patch.object(fp.posterior_profile_plotter, "get2DDensityGridData") as mock_get2d:
            profiler.precompute_1D(["p0", "p1"], num_points_1D=8)
            profiler.precompute_2D([("p0", "p1")], num_points_2D=8)
        self.assertEqual(mock_get1d.call_count, 2)
        mock_get2d.assert_called_once_with("p0", "p1", num_points_2D=8)

    def test_normalize(self):
        """normalize rescales all the cached profiles."""
        profiler = _profiler()
        x = np.linspace(-3.0, 3.0, 61)
        profiler.profile_density_1D = {0: Density1D(x, 3.0 * np.exp(-0.5 * x**2))}
        density_2D = Density2D(x, x, 2.0 * np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2)))
        profiler.profile_density_2D = {0: {1: density_2D}, 1: {}}
        profiler.normalize(by="max")
        self.assertAlmostEqual(np.max(profiler.profile_density_1D[0].P), 1.0)
        self.assertAlmostEqual(np.max(profiler.profile_density_2D[0][1].P), 1.0)
        profiler.normalize(by="integral")
        self.assertAlmostEqual(profiler.profile_density_1D[0].norm_integral(), 1.0)
        self.assertAlmostEqual(profiler.profile_density_2D[0][1].norm_integral(), 1.0)

    def test_profile_variance_requires_average_flow(self):
        """get_1d_profile_variance needs an average flow."""
        with self.assertRaises(ValueError):
            _profiler().get_1d_profile_variance("p0")

    def test_scipy_polish_bounds_follow_parameter_order(self):
        """Scipy polishing bounds follow the parameter order, not the ranges dict order."""
        flow = StubFlow(num_params=3)
        flow.parameter_ranges = {"p2": (-3.0, 3.0), "p0": (-1.0, 1.0), "p1": (-2.0, 2.0)}
        profiler = _profiler(flow, use_scipy=True, pre_polish=False, polish=True, smoothing=False,
                             num_minimization_samples=50)
        with patch("tensiometer.synthetic_probability.flow_profiler.minimize", wraps=scipy.optimize.minimize) as mock_min:
            profiler.get1DDensityGridData("p0", num_points_1D=4)
        self.assertGreater(mock_min.call_count, 0)
        self.assertEqual(list(mock_min.call_args.kwargs["bounds"]), [(-2.0, 2.0), (-3.0, 3.0)])
        mock_min.reset_mock()
        with patch("tensiometer.synthetic_probability.flow_profiler.minimize", wraps=scipy.optimize.minimize) as mock_min:
            profiler.get2DDensityGridData("p0", "p2", num_points_2D=4)
        self.assertGreater(mock_min.call_count, 0)
        self.assertEqual(list(mock_min.call_args.kwargs["bounds"]), [(-2.0, 2.0)])

    def test_torch_polish_accepts_only_improvements(self):
        """_torch_polish keeps the fixed columns and updates only improved points."""
        profiler = _profiler(StubFlow(num_params=3))
        profiler.temp_inv_cov = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 4.0]])
        samples = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        logP = np.array([-1.0, -1.0])
        fake_result = _fake_minimize_result(np.array([[9.0, 8.0], [7.0, 6.0]]), np.array([0.5, 2.0]))
        with patch("tensiometer.synthetic_probability.flow_profiler.batched_minimize",
                   return_value=fake_result) as mock_min:
            new_logP, new_samples = profiler._torch_polish(samples, logP, [1], np.array([1.0, 0.0, 1.0]))
        self.assertIsInstance(new_logP, np.ndarray)
        self.assertIsInstance(new_samples, np.ndarray)
        self.assertTrue(np.allclose(new_logP, [-0.5, -1.0]))
        self.assertTrue(np.allclose(new_samples, [[9.0, 0.2, 8.0], [0.4, 0.5, 0.6]]))
        args, kwargs = mock_min.call_args
        self.assertTrue(np.allclose(tu.to_numpy(args[1]), samples[:, [0, 2]]))
        expected_hessian = np.linalg.inv(np.array([[2.0, 0.0], [0.0, 4.0]]))
        self.assertTrue(np.allclose(tu.to_numpy(kwargs["initial_inverse_hessian"]), expected_hessian))
        self.assertEqual(kwargs["max_iterations"], profiler.options["torch_max_iterations"])

    def test_torch_polish_singular_fisher(self):
        """A singular masked Fisher matrix falls back to the identity."""
        profiler = _profiler()
        profiler.temp_inv_cov = np.zeros((2, 2))
        fake_result = _fake_minimize_result(np.array([[0.0]]), 1.0)
        with patch("tensiometer.synthetic_probability.flow_profiler.batched_minimize",
                   return_value=fake_result) as mock_min:
            profiler._torch_polish(np.array([[0.1, 0.2]]), np.array([0.0]), [0], np.array([0.0, 1.0]))
        self.assertTrue(np.allclose(tu.to_numpy(mock_min.call_args.kwargs["initial_inverse_hessian"]), np.eye(1)))

    def test_torch_polish_empty(self):
        """An empty population is returned unchanged."""
        profiler = _profiler()
        logP, samples = profiler._torch_polish(np.zeros((0, 2)), np.zeros(0), [0], np.array([0.0, 1.0]))
        self.assertEqual(samples.shape, (0, 2))
        self.assertEqual(logP.shape, (0,))

    def test_torch_polish_real_minimizer_box_prior(self):
        """_torch_polish maximizes the free coordinates inside the box."""
        mean = np.array([0.5, 0.6])
        precision_matrix = np.array([[4.0, 1.0], [1.0, 2.0]])
        flow = GaussianStubFlow(mean, precision_matrix, lower=0.0, upper=1.0)
        profiler = _profiler(flow, box_prior=True, use_scipy=False)
        profiler.temp_inv_cov = precision_matrix
        samples = np.array([[0.2, 0.3], [0.7, 0.9]])
        logP = tu.to_numpy(flow.log_probability(samples))
        new_logP, new_samples = profiler._torch_polish(samples, logP, [0], np.array([0.0, 1.0]))
        conditional = mean[1] - precision_matrix[1, 0] / precision_matrix[1, 1] * (samples[:, 0] - mean[0])
        self.assertTrue(np.allclose(new_samples[:, 0], samples[:, 0]))
        self.assertTrue(np.allclose(new_samples[:, 1], conditional, atol=1e-3))
        self.assertTrue(np.all(new_logP > logP))

    def test_masked_gradient_ascent_no_iterations(self):
        """Zero iterations return the input as numpy with python integers."""
        profiler = _profiler()
        profiler.temp_inv_cov = np.eye(2)
        ensemble, logP, num_moving, num_iter = profiler._masked_gradient_ascent(
            learning_rate=0.1, num_iterations=0, ensemble=np.zeros((1, 2)), mask=np.array([1.0, 0.0]))
        self.assertIsInstance(ensemble, np.ndarray)
        self.assertIsInstance(logP, np.ndarray)
        self.assertIsInstance(num_moving, int)
        self.assertIsInstance(num_iter, int)
        self.assertEqual(ensemble.shape, (1, 2))
        self.assertEqual(num_iter, 0)

    def test_masked_gradient_ascent_box_prior(self):
        """With a box prior the ascent moves the unmasked coordinates inside the box."""
        mean = np.array([0.5, 0.5, 0.5])
        precision_matrix = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 3.0]])
        flow = GaussianStubFlow(mean, precision_matrix, lower=0.0, upper=1.0)
        profiler = _profiler(flow, box_prior=True)
        profiler.temp_inv_cov = precision_matrix
        ensemble = np.array([[0.1, 0.9, 0.2], [0.8, 0.3, 0.95]])
        initial_logP = tu.to_numpy(flow.log_probability(ensemble))
        mask = np.array([1.0, 0.0, 1.0])
        new_ensemble, logP, num_moving, num_iter = profiler._masked_gradient_ascent(
            learning_rate=0.1, num_iterations=50, ensemble=ensemble, mask=mask)
        self.assertTrue(np.allclose(new_ensemble[:, 1], ensemble[:, 1], atol=1e-5))
        self.assertTrue(np.all((new_ensemble > 0.0) & (new_ensemble < 1.0)))
        self.assertTrue(np.all(logP > initial_logP))
        self.assertTrue(np.allclose(logP, tu.to_numpy(flow.log_probability(new_ensemble)), atol=1e-5))
        self.assertLessEqual(num_iter, 50)

#########################################################################################################
# Iterative cache


class TestProfilerIterative(unittest.TestCase):
    """update_cache_iterative with a fixed population."""

    def setUp(self):
        """Stub the minimizers."""
        self._points_patch = patch(
            "tensiometer.synthetic_probability.flow_profiler.points_minimizer",
            return_value=(np.array([True]), np.array([0.0]), np.array([[0.0, 0.0]])))
        self._points_patch.start()
        self._torch_patch = patch(
            "tensiometer.synthetic_probability.flow_profiler.batched_minimize",
            return_value=_fake_minimize_result(np.zeros((1, 2)), 0.0))
        self._torch_patch.start()

    def tearDown(self):
        """Remove the patches."""
        self._torch_patch.stop()
        self._points_patch.stop()

    def _run_iterative(self, profiler):
        """Run one iteration with stubbed marginal densities on the grid [0, 0.5, 1]."""
        dummy1d = types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]))
        dummy2d = types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]), y=np.array([0.0, 0.5, 1.0]))
        with patch("getdist.mcsamples.MCSamples.get1DDensityGridData", return_value=dummy1d), \
                patch("getdist.mcsamples.MCSamples.get2DDensityGridData", return_value=dummy2d), _quiet():
            profiler.update_cache_iterative(params=["p0", "p1"], niter=1, update_1D=True, update_2D=True)

    def test_update_cache_iterative(self):
        """The best point of every bin is stored."""
        profiler = StubProfiler(StubFlow(lower=0.0, upper=1.0))
        self._run_iterative(profiler)
        self.assertTrue(np.allclose(profiler._1d_logP["p0"], [0.1, 0.2]))
        self.assertTrue(np.allclose(profiler._1d_logP["p1"], [0.1, 0.2]))
        self.assertTrue(np.allclose(profiler._1d_samples["p0"], StubProfiler.population))
        logP_2D = profiler._2d_logP["p0", "p1"]
        self.assertTrue(np.allclose(np.diag(logP_2D), [0.1, 0.2]))
        self.assertTrue(np.all(np.isneginf([logP_2D[0, 1], logP_2D[1, 0]])))
        self.assertTrue(np.allclose(profiler._2d_samples["p0", "p1"][1, 1], StubProfiler.population[1]))
        self.assertTrue(np.all(np.isnan(profiler._2d_samples["p0", "p1"][0, 1])))
        self.assertIsNone(profiler.temp_samples)
        profiler.reset_cache()
        self.assertEqual(profiler.profile_density_1D, {})

    def test_update_cache_iterative_logP_from_filtered_samples(self):
        """The logP of the best points comes from the samples inside the bins."""

        class OutsideProfiler(StubProfiler):
            """Profiler whose first sample is outside the bins and has a large logP."""

            population = np.array([[-5.0, -5.0], [0.2, 0.3], [0.6, 0.7]])
            population_logP = np.array([9.0, 0.1, 0.2])

        profiler = OutsideProfiler(StubFlow(lower=0.0, upper=1.0))
        self._run_iterative(profiler)
        self.assertTrue(np.allclose(profiler._1d_logP["p0"], [0.1, 0.2]))
        self.assertTrue(np.allclose(np.diag(profiler._2d_logP["p0", "p1"]), [0.1, 0.2]))

    def test_find_MAP_x0_abstract(self):
        """find_MAP from x0 in abstract coordinates with the torch branch."""
        flow = StubFlow(with_ranges=False)
        profiler = _profiler(flow, use_scipy=False, pre_polish=False, polish=False, smoothing=False)
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   side_effect=fp.points_minimizer):
            value, solution = profiler.find_MAP(x0=np.zeros((1, flow.num_params)), randomize=False, abstract=True)
        self.assertEqual(solution.shape, (flow.num_params,))
        self.assertEqual(value, 0.0)

    def test_update_cache_iterative_without_ranges_feedback(self):
        """Without parameter ranges the full marginal grids are used; feedback prints the progress."""
        profiler = StubProfiler(StubFlow(with_ranges=False))
        profiler.feedback = 1
        grid = np.array([-0.5, 0.5, 1.5])
        dummy1d = types.SimpleNamespace(x=grid)
        dummy2d = types.SimpleNamespace(x=grid, y=grid)
        buffer = io.StringIO()
        with patch("getdist.mcsamples.MCSamples.get1DDensityGridData", return_value=dummy1d), \
                patch("getdist.mcsamples.MCSamples.get2DDensityGridData", return_value=dummy2d), \
                contextlib.redirect_stdout(buffer):
            profiler.update_cache_iterative(niter=1, update_1D=True, update_2D=True)
        output = buffer.getvalue()
        self.assertIn("initializing profiler data", output)
        self.assertIn("parameters: ['p0', 'p1']", output)
        self.assertIn("parameter pairs: [('p0', 'p1')]", output)
        self.assertIn("  * initializing 1D profiles", output)
        self.assertIn("  * initializing 2D profiles", output)
        self.assertTrue(np.array_equal(profiler._1d_bins["p0"], grid))
        x_bins, y_bins = profiler._2d_bins["p0", "p1"]
        self.assertTrue(np.array_equal(x_bins, grid))
        self.assertTrue(np.array_equal(y_bins, grid))
        self.assertTrue(np.allclose(profiler._1d_logP["p0"], [0.1, 0.2]))
        self.assertTrue(np.allclose(profiler._2d_logP["p0", "p1"][0, 0], 0.1))
        self.assertTrue(np.allclose(profiler._2d_logP["p0", "p1"][1, 1], 0.2))

#########################################################################################################
# Feedback and less common branches


class TestProfilerFeedbackAndBranches(unittest.TestCase):
    """Feedback output and the less common branches of the profiler."""

    def test_points_minimizer_scipy_reports_failure(self):
        """With feedback the scipy branch prints the message of failed minimizations."""
        fake_res = types.SimpleNamespace(success=False, fun=2.0, x=np.array([0.5]), message="line search failed")
        buffer = io.StringIO()
        with patch(FLOW_PROFILER + "minimize", return_value=fake_res), contextlib.redirect_stdout(buffer):
            success, value, point = fp.points_minimizer(
                lambda x: np.sum(x**2), lambda x: 2 * x, np.array([[1.0]]), feedback=2)
        self.assertFalse(success[0])
        self.assertEqual(value[0], 2.0)
        self.assertTrue(np.allclose(point, [[0.5]]))
        self.assertIn("Success False", buffer.getvalue())
        self.assertIn("line search failed", buffer.getvalue())

    def test_find_flow_MAP_default_num_samples(self):
        """The default random search draws 1000 samples per parameter."""
        flow = StubFlow(num_params=3)
        fake_res = (np.array([True]), np.array([0.0]), np.zeros((1, 3)))
        with patch.object(flow, "sample", wraps=flow.sample) as mock_sample, \
                patch(FLOW_PROFILER + "points_minimizer", return_value=fake_res) as mock_pm:
            fp.find_flow_MAP(flow, feedback=0, abstract=False)
        mock_sample.assert_called_once_with(3000)
        self.assertEqual(mock_pm.call_args.args[2].shape, (10, 3))

    def test_initialize_cache_calls_update_cache(self):
        """With initialize_cache the constructor fills the caches with its keyword arguments."""
        with patch.object(fp.posterior_profile_plotter, "update_cache") as mock_update:
            profiler = fp.posterior_profile_plotter(StubFlow(), feedback=0, initialize_cache=True, num_points_1D=8)
        mock_update.assert_called_once_with(initialize_cache=True, num_points_1D=8)
        self.assertEqual(profiler.options["num_points_1D"], 8)

    def test_update_cache_feedback(self):
        """update_cache finds the MAP and all the 1D and 2D profiles, printing the progress."""
        profiler = StubProfiler(StubFlow(num_params=3))
        profiler.feedback = 1
        buffer = io.StringIO()
        with patch.object(profiler, "find_MAP") as mock_map, \
                patch.object(profiler, "get1DDensityGridData") as mock_1d, \
                patch.object(profiler, "get2DDensityGridData") as mock_2d, \
                contextlib.redirect_stdout(buffer):
            profiler.update_cache(num_points_1D=8)
        output = buffer.getvalue()
        for message in ("initializing profiler data", "finding MAP", "initializing 1D profiles",
                        "initializing 2D profiles"):
            self.assertIn(message, output)
        mock_map.assert_called_once_with(num_points_1D=8)
        self.assertEqual([call.args[0] for call in mock_1d.call_args_list], [0, 1, 2])
        self.assertEqual([call.args[:2] for call in mock_2d.call_args_list], [(0, 1), (0, 2), (1, 2)])
        self.assertIsNone(profiler.temp_samples)
        self.assertIsNone(profiler.temp_probs)

    def test_find_MAP_feedback(self):
        """find_MAP samples the population when missing and reports the minimizer in use."""
        fake_res = (np.array([True]), np.array([-0.5]), np.array([[0.1, 0.2]]))
        for use_scipy, label in ((True, "(scipy)"), (False, "(torch)")):
            with self.subTest(use_scipy=use_scipy):
                profiler = _profiler(use_scipy=use_scipy)
                profiler.feedback = 2
                buffer = io.StringIO()
                with patch(FLOW_PROFILER + "points_minimizer", return_value=fake_res) as mock_pm, \
                        contextlib.redirect_stdout(buffer):
                    value, solution = profiler.find_MAP(num_best_to_follow=2, num_minimization_samples=20)
                output = buffer.getvalue()
                self.assertIn("finding global best fit", output)
                self.assertIn("doing initial randomized search", output)
                self.assertIn("time taken for random initial selection", output)
                self.assertIn("doing minimization " + label, output)
                self.assertIn("time taken for minimization", output)
                self.assertEqual(profiler.temp_samples.shape, (20, 2))
                self.assertEqual(mock_pm.call_args.args[2].shape, (2, 2))
                self.assertEqual(value, 0.5)
                self.assertTrue(np.allclose(solution, [0.1, 0.2]))

    def test_1D_profile_flat_flow_scipy_feedback(self):
        """On a flat flow no polishing step improves the random search points, and feedback reports it."""
        profiler = _profiler(StubFlow(with_ranges=False), use_scipy=True, pre_polish=True, polish=True,
                             smoothing=True, num_minimization_samples=500)
        profiler.feedback = 3
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), np.errstate(invalid="ignore"):
            density = profiler.get1DDensityGridData("p1", num_points_1D=8, scipy_method="BFGS",
                                                    scipy_options={"maxiter": 0}, scipy_use_jac=False,
                                                    num_gd_interactions_1D=5)
        output = buffer.getvalue()
        for message in ("calculating the 1D profile for: p1 with index 1", "doing initial randomized search",
                        "number of empty/filled 1D bins", "doing gradient descent pre-polishing",
                        "gradient descents did not improve", "doing minimization polishing (scipy)",
                        "method: BFGS", "use_jac: False", "Maximum number of iterations",
                        "Warning minimization failed", "time taken for polishing",
                        "doing interpolation on regular grid", "smoothing results", "time taken for smoothing"):
            self.assertIn(message, output)
        self.assertEqual(len(density.x), 9)
        self.assertAlmostEqual(np.max(density.P), 1.0)
        self.assertGreater(density.profile_smoothing_scale_1D, 0.0)
        self.assertTrue(_is_subset_of(density.profile_subspace, profiler.temp_samples))

    def test_2D_profile_flat_flow_scipy_feedback_smoothing(self):
        """On a flat flow the 2D profile keeps the random search points and is smoothed on the grid."""
        profiler = _profiler(StubFlow(num_params=3, with_ranges=False), use_scipy=True, pre_polish=True,
                             polish=True, smoothing=True, num_minimization_samples=500)
        profiler.feedback = 3
        buffer = io.StringIO()
        with patch(FLOW_PROFILER + "gaussian_filter", wraps=fp.gaussian_filter) as mock_filter, \
                contextlib.redirect_stdout(buffer), np.errstate(invalid="ignore"):
            density = profiler.get2DDensityGridData("p0", "p2", num_points_2D=6, scipy_method="BFGS",
                                                    scipy_options={"maxiter": 0}, scipy_use_jac=False,
                                                    num_gd_interactions_2D=5, smooth_scale_2D=0.0)
        output = buffer.getvalue()
        for message in ("calculating the 2D profile for: p0, p2", "doing initial randomized search",
                        "number of empty/filled 2D bins", "doing gradient descent pre-polishing",
                        "gradient descents did not improve", "doing minimization polishing (scipy)",
                        "method: BFGS", "use_jac: False", "Maximum number of iterations",
                        "Warning minimization failed", "time taken for polishing",
                        "doing interpolation on regular grid", "smoothing results", "time taken for smoothing"):
            self.assertIn(message, output)
        self.assertEqual(density.P.shape, (6, 6))
        self.assertAlmostEqual(np.max(density.P), 1.0)
        self.assertEqual(profiler.profile_density_2D[2][0].P.shape, (6, 6))
        self.assertTrue(_is_subset_of(density.profile_subspace, profiler.temp_samples))
        expected_sigmas = []
        for idx, grid in ((0, density.x), (2, density.y)):
            error = profiler._initParamRanges(idx, None).err
            expected_sigmas.append(0.2 * len(grid) / (grid[-1] - grid[0]) * error)
        mock_filter.assert_called_once()
        self.assertTrue(np.allclose(sorted(mock_filter.call_args.args[1]), sorted(expected_sigmas)))

    def test_2D_profile_index_order_scipy(self):
        """The scipy polish reinserts the fixed coordinates in place, whatever the order of the two indexes."""
        for name1, name2, idx1, idx2 in (("p2", "p0", 2, 0), ("p0", "p2", 0, 2)):
            with self.subTest(name1=name1, name2=name2):
                profiler = _profiler(_gaussian_stub_3D(), use_scipy=True, pre_polish=False, polish=True,
                                     smoothing=False, num_minimization_samples=500)
                density = profiler.get2DDensityGridData(name1, name2, num_points_2D=4)
                subspace = density.profile_subspace
                self.assertEqual(density.P.shape, (4, 4))
                self.assertIs(profiler.profile_density_2D[idx1][idx2], density)
                self.assertTrue(np.allclose(subspace[:, 1], _conditional_maximum(subspace, 1), atol=1e-3))

    def test_profiles_torch_polish_feedback(self):
        """The torch polish of the profiles keeps the profiled coordinates fixed."""
        flow = _gaussian_stub_3D()
        profiler = _profiler(flow, use_scipy=False, pre_polish=False, polish=True, smoothing=False,
                             num_minimization_samples=500)
        profiler.feedback = 2
        buffer = io.StringIO()
        with patch.object(profiler, "_torch_polish", wraps=profiler._torch_polish) as mock_polish, \
                contextlib.redirect_stdout(buffer):
            profiler.get1DDensityGridData("p1", num_points_1D=8)
            density = profiler.get2DDensityGridData("p0", "p2", num_points_2D=4)
        self.assertEqual(buffer.getvalue().count("doing minimization polishing (torch)"), 2)
        first_call, second_call = mock_polish.call_args_list
        self.assertEqual(first_call.args[2], [1])
        self.assertTrue(np.array_equal(first_call.args[3], [1.0, 0.0, 1.0]))
        self.assertEqual(second_call.args[2], [0, 2])
        self.assertTrue(np.array_equal(second_call.args[3], [0.0, 1.0, 0.0]))
        subspace = density.profile_subspace
        self.assertTrue(np.allclose(subspace[:, 1], _conditional_maximum(subspace, 1), atol=1e-3))

    def test_torch_polish_reports_unconverged_points(self):
        """_torch_polish reports the points that did not converge and still keeps the improved ones."""
        profiler = _profiler()
        profiler.feedback = 2
        profiler.temp_inv_cov = np.eye(2)
        fake_result = _fake_minimize_result(np.array([[0.5], [0.6]]), np.array([-1.0, 3.0]),
                                            converged=np.array([False, True]))
        buffer = io.StringIO()
        with patch(FLOW_PROFILER + "batched_minimize", return_value=fake_result), contextlib.redirect_stdout(buffer):
            new_logP, new_samples = profiler._torch_polish(
                np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.0, 0.0]), [0], np.array([0.0, 1.0]))
        self.assertIn("Warning minimization failed for  1 points", buffer.getvalue())
        self.assertIn("but  1 samples were still better.", buffer.getvalue())
        self.assertTrue(np.allclose(new_logP, [1.0, 0.0]))
        self.assertTrue(np.allclose(new_samples, [[0.1, 0.5], [0.3, 0.4]]))

#########################################################################################################
# Regressions of fixed bugs


class TestProfilerRegressions(unittest.TestCase):
    """Regression tests for fixed ``flow_profiler`` bugs."""

    def test_1D_unknown_parameter_returns_none(self):
        """Like getdist, an unknown parameter has no 1D profile."""
        profiler = _profiler(num_minimization_samples=50)
        self.assertIsNone(profiler.get1DDensityGridData("missing"))

    def test_2D_unknown_parameter_returns_none(self):
        """Like getdist, an unknown parameter has no 2D profile."""
        profiler = _profiler(num_minimization_samples=50)
        self.assertIsNone(profiler.get2DDensityGridData("p0", "missing"))

    def test_1D_random_search_keeps_first_sample(self):
        """The best sample of every bin is kept, including the first sample."""
        profiler = _profiler(pre_polish=False, polish=False, smoothing=False)
        profiler.temp_samples = np.array([[0.1, 0.0], [0.3, 0.0], [0.5, 0.0], [0.7, 0.0], [0.9, 0.0]],
                                         dtype=tu.np_prec)
        profiler.temp_probs = np.zeros(5, dtype=tu.np_prec)
        grid = types.SimpleNamespace(x=np.linspace(0.0, 1.0, 6), view_ranges=[0.0, 1.0])
        with patch("getdist.mcsamples.MCSamples.get1DDensityGridData", return_value=grid):
            density = profiler.get1DDensityGridData("p0", num_points_1D=5)
        self.assertEqual(len(density.profile_subspace), 5)

    def test_2D_random_search_keeps_first_sample(self):
        """The best sample of every 2D bin is kept, including the first sample."""
        profiler = _profiler(StubFlow(num_params=3), pre_polish=False, polish=False, smoothing=False)
        centres = np.array([1.0, 3.0, 5.0]) / 6.0
        population = np.array([[x, y, 0.0] for x in centres for y in centres], dtype=tu.np_prec)
        profiler.temp_samples = population
        profiler.temp_probs = np.zeros(len(population), dtype=tu.np_prec)
        grid = types.SimpleNamespace(x=np.linspace(0.0, 1.0, 4), y=np.linspace(0.0, 1.0, 4),
                                     view_ranges=[[0.0, 1.0], [0.0, 1.0]])
        with patch("getdist.mcsamples.MCSamples.get2DDensityGridData", return_value=grid):
            density = profiler.get2DDensityGridData("p0", "p1", num_points_2D=3)
        self.assertEqual(len(density.profile_subspace), 9)

    def test_2D_profile_large_log_probability(self):
        """As in 1D, the 2D profile is computed relative to the maximum log probability."""
        densities = {}
        for offset in (0.0, 100.0, -120.0):
            torch.manual_seed(0)
            profiler = _profiler(OffsetGaussianStubFlow(offset), use_scipy=True, pre_polish=False, polish=False,
                                 smoothing=False, num_minimization_samples=500)
            densities[offset] = profiler.get2DDensityGridData("p0", "p2", num_points_2D=6)
        for offset in (100.0, -120.0):
            self.assertTrue(np.all(np.isfinite(densities[offset].P)))
            self.assertTrue(np.allclose(densities[offset].P, densities[0.0].P, atol=1e-4))
            self.assertAlmostEqual(densities[offset].maximum - densities[0.0].maximum, offset, places=3)

    def test_1D_profile_of_one_parameter_flow(self):
        """With a single parameter the 1D profile is the density, evaluated directly on the grid."""
        profiler = _profiler(GaussianStubFlow([0.1], np.array([[4.0]])), num_minimization_samples=100)
        density = profiler.get1DDensityGridData("p0", num_points_1D=16)
        log_P = -2.0 * (density.x - 0.1)**2
        self.assertTrue(np.allclose(density.P, np.exp(log_P - np.max(log_P)), atol=1e-5))
        self.assertAlmostEqual(density.maximum, np.max(log_P), places=5)
        self.assertEqual(density.profile_subspace.shape, (17, 1))
        self.assertIsNone(density.profile_smoothing_scale_1D)
        self.assertIs(profiler.profile_density_1D[0], density)

    def test_profile_on_grid_non_finite_log_probability(self):
        """On the grid, non finite log probabilities have zero probability, and none finite is an error."""
        profiler = _profiler(NonFiniteGaussianStubFlow(0.0), num_minimization_samples=100)
        density = profiler.get1DDensityGridData("p0", num_points_1D=16)
        self.assertTrue(np.any(density.x > 0.0))
        self.assertTrue(np.all(density.P[density.x > 0.0] == 0.0))
        self.assertTrue(np.all(density.P[density.x <= 0.0] > 0.0))
        profiler = _profiler(NonFiniteGaussianStubFlow(-10.0), num_minimization_samples=100)
        with self.assertRaises(ValueError):
            profiler.get1DDensityGridData("p0", num_points_1D=16)

    def test_2D_profile_of_two_parameter_flow(self):
        """With two parameters the 2D profile is the density, evaluated directly on the grid."""
        mean = np.array([0.1, -0.1])
        precision = np.array([[4.0, 1.0], [1.0, 2.0]])
        for name1, name2, idx1, idx2 in (("p0", "p1", 0, 1), ("p1", "p0", 1, 0)):
            with self.subTest(name1=name1, name2=name2):
                profiler = _profiler(GaussianStubFlow(mean, precision), num_minimization_samples=100)
                density = profiler.get2DDensityGridData(name1, name2, num_points_2D=8)
                mesh_x, mesh_y = np.meshgrid(density.x, density.y)
                points = np.zeros(mesh_x.shape + (2,))
                points[..., idx1] = mesh_x
                points[..., idx2] = mesh_y
                diff = points - mean
                log_P = -0.5 * np.einsum("...i,ij,...j->...", diff, precision, diff)
                self.assertTrue(np.allclose(density.P, np.exp(log_P - np.max(log_P)), atol=1e-5))
                self.assertAlmostEqual(density.maximum, np.max(log_P), places=5)
                self.assertEqual(density.profile_subspace.shape, (64, 2))
                transposed = profiler.get2DDensityGridData(name2, name1, num_points_2D=8)
                self.assertTrue(np.allclose(transposed.P, density.P.T))

    def test_2D_smoothing_scales_follow_the_grid_axes(self):
        """Axis 0 of the 2D grid is y (getdist ``P[y, x]``), so its smoothing scale must come from the y parameter."""
        flow = StubFlow(num_params=3, with_ranges=False)
        flow.chain_samples[:, 2] = np.random.default_rng(1).uniform(-1.0, 1.0, len(flow.chain_samples))
        profiler = _profiler(flow, use_scipy=True, pre_polish=False, polish=False, smoothing=True,
                             num_minimization_samples=500)
        with patch(FLOW_PROFILER + "gaussian_filter", wraps=fp.gaussian_filter) as mock_filter:
            density = profiler.get2DDensityGridData("p0", "p2", num_points_2D=6, smooth_scale_2D=0.5)
        sigma_x = 0.5 * 6 / (density.x[-1] - density.x[0]) * profiler._initParamRanges(0, None).err
        sigma_y = 0.5 * 6 / (density.y[-1] - density.y[0]) * profiler._initParamRanges(2, None).err
        self.assertTrue(np.allclose(mock_filter.call_args.args[1], [sigma_y, sigma_x]))

#########################################################################################################
# Real Gaussian flow


def _gaussian_chain(seed, num_samples=1000):
    """Seeded correlated 3D Gaussian chain with loglikes.

    :param seed: random seed
    :param num_samples: number of samples
    :returns: the samples ``(N, 3)`` and the :class:`getdist.MCSamples` chain
    """
    mean = np.array([0.5, -1.0, 2.0])
    cov = np.array([[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 0.5]])
    samples = np.random.default_rng(seed).multivariate_normal(mean, cov, size=num_samples)
    diff = samples - mean
    loglikes = 0.5 * np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)
    chain = MCSamples(samples=samples, loglikes=loglikes, names=["a", "b", "c"], labels=["A", "B", "C"],
                      settings=GETDIST_SETTINGS)
    return samples, chain


def _gaussian_flow(seed):
    """Exactly Gaussian flow (no trainable bijector, no prior bijector) on a seeded chain."""
    samples, chain = _gaussian_chain(seed)
    flow = sp.FlowCallback(chain, trainable_bijector=None, prior_bijector=None, feedback=0)
    return samples, flow


class TestGaussianFlowProfiler(unittest.TestCase):
    """Profiles of an exactly Gaussian flow match the analytic results."""

    @classmethod
    def setUpClass(cls):
        """Build the flow."""
        samples, cls.flow = _gaussian_flow(0)
        cls.mean = samples.mean(axis=0)
        cls.cov = np.cov(samples.T)
        cls.sigma = np.sqrt(np.diag(cls.cov))

    def _profiler(self, use_scipy, **kwargs):
        """Seeded profiler of the Gaussian flow."""
        torch.manual_seed(0)
        return fp.posterior_profile_plotter(self.flow, feedback=0, use_scipy=use_scipy, smoothing=False,
                                            num_minimization_samples=2000, **kwargs)

    def _max_logP(self):
        """Peak log density of the Gaussian flow."""
        return float(tu.to_numpy(self.flow.log_probability(self.mean[None, :]))[0])

    def test_flow_is_gaussian(self):
        """The flow log probability is the Gaussian one."""
        points = self.mean + np.array([[0.0, 0.0, 0.0], [0.5, -0.3, 0.2]])
        expected = scipy.stats.multivariate_normal(self.mean, self.cov).logpdf(points)
        logP = self.flow.log_probability(points)
        self.assertEqual(logP.device.type, "cpu")
        self.assertEqual(logP.dtype, tu.get_precision())
        self.assertTrue(np.allclose(tu.to_numpy(logP), expected, atol=1e-2))

    def test_MAP_matches_mean(self):
        """The MAP is the Gaussian mean, for all the MAP strategies and both minimizers."""
        for use_scipy in (True, False):
            for mode in ("random", "abstract", "chain", "box"):
                with self.subTest(use_scipy=use_scipy, mode=mode):
                    profiler = self._profiler(use_scipy, box_prior=(mode == "box"))
                    with _quiet():
                        if mode == "chain":
                            value, solution = profiler.find_MAP(randomize=False, num_best_to_follow=5)
                        else:
                            value, solution = profiler.find_MAP(num_best_to_follow=5, abstract=(mode == "abstract"))
                    self.assertTrue(np.all(np.abs(solution - self.mean) < 5e-3 * self.sigma))
                    self.assertAlmostEqual(value, self._max_logP(), places=3)
                    bestfit = profiler.getBestFit()
                    self.assertTrue(np.allclose([par.best_fit for par in bestfit.names], solution))

    def test_1D_profile_peaks_at_mean(self):
        """The 1D profile of a Gaussian peaks at the mean and follows the conditional maximum."""
        num_bins = 16
        for use_scipy in (True, False):
            with self.subTest(use_scipy=use_scipy):
                profiler = self._profiler(use_scipy)
                density = profiler.get1DDensityGridData("a", num_points_1D=num_bins)
                spacing = density.x[1] - density.x[0]
                self.assertEqual(len(density.x), num_bins + 1)
                self.assertAlmostEqual(np.max(density.P), 1.0)
                self.assertLessEqual(abs(density.x[np.argmax(density.P)] - self.mean[0]), spacing)
                # the profiled coordinates are at the conditional maximum:
                subspace = density.profile_subspace
                slope = self.cov[1:, 0] / self.cov[0, 0]
                conditional = self.mean[1:] + np.outer(subspace[:, 0] - self.mean[0], slope)
                self.assertTrue(np.allclose(subspace[:, 1:], conditional, atol=1e-2))
                # the profile of a Gaussian is a Gaussian with the marginal variance:
                inside = (density.x > subspace[:, 0].min()) & (density.x < subspace[:, 0].max())
                analytic = np.exp(-0.5 * (density.x - self.mean[0])**2 / self.cov[0, 0])
                self.assertTrue(np.allclose(density.P[inside], analytic[inside], atol=5e-2))
                self.assertIs(profiler.get1DDensityGridData("a"), density)

    def test_2D_profile_peaks_at_mean(self):
        """The 2D profile of a Gaussian peaks at the mean and follows the conditional maximum."""
        num_bins = 8
        for use_scipy in (True, False):
            with self.subTest(use_scipy=use_scipy):
                profiler = self._profiler(use_scipy)
                density = profiler.get2DDensityGridData("a", "b", num_points_2D=num_bins)
                self.assertEqual(density.P.shape, (num_bins, num_bins))
                self.assertAlmostEqual(np.max(density.P), 1.0)
                self.assertEqual(profiler.profile_density_2D[1][0].P.shape, density.P.T.shape)
                subspace = density.profile_subspace
                weights = np.linalg.solve(self.cov[:2, :2], self.cov[:2, 2])
                conditional = self.mean[2] + (subspace[:, :2] - self.mean[:2]) @ weights
                self.assertTrue(np.allclose(subspace[:, 2], conditional, atol=1e-2))
                iy, ix = np.unravel_index(np.argmax(density.P), density.P.shape)
                self.assertLessEqual(abs(density.x[ix] - self.mean[0]), density.x[1] - density.x[0])
                self.assertLessEqual(abs(density.y[iy] - self.mean[1]), density.y[1] - density.y[0])

    def test_update_cache_likestats_normalize(self):
        """update_cache fills the caches; likestats bracket the MAP; normalize works on real profiles."""
        profiler = self._profiler(False)
        with _quiet():
            profiler.update_cache(params=["a", "b"], num_points_1D=16, num_points_2D=8, num_best_to_follow=5)
        self.assertIsNone(profiler.temp_samples)
        self.assertEqual(sorted(profiler.profile_density_1D.keys()), [0, 1])
        self.assertIn(1, profiler.profile_density_2D[0])
        self.assertTrue(np.all(np.abs(profiler.flow_MAP - self.mean) < 5e-3 * self.sigma))
        profiler.get1DDensityGridData("c", num_points_1D=16)
        stats = profiler.getLikeStats(profile_lims=True)
        self.assertIsNotNone(stats)
        for j, par in enumerate(profiler.paramNames.names):
            self.assertTrue(np.all(par.ND_limit_bot < profiler.flow_MAP[j]))
            self.assertTrue(np.all(par.ND_limit_top > profiler.flow_MAP[j]))
        profiler.normalize(by="integral")
        self.assertAlmostEqual(profiler.profile_density_1D[0].norm_integral(), 1.0, places=6)
        profiler.normalize(by="max")
        self.assertAlmostEqual(np.max(profiler.profile_density_2D[0][1].P), 1.0)

    def test_update_cache_iterative(self):
        """The iterative cache gives profiles that peak at the mean."""
        profiler = self._profiler(False)
        with _quiet():
            profiler.update_cache_iterative(params=["a", "b"], niter=2, update_1D=True, update_2D=True,
                                            num_points_1D=16, num_points_2D=8)
        self.assertEqual(sorted(profiler.profile_density_1D.keys()), [0, 1])
        self.assertIn(1, profiler.profile_density_2D[0])
        for idx in (0, 1):
            density = profiler.profile_density_1D[idx]
            spacing = density.x[1] - density.x[0]
            self.assertLessEqual(abs(density.x[np.argmax(density.P)] - self.mean[idx]), spacing)

    def test_update_cache_iterative_other_parameters_and_reset(self):
        """Parameters the iterative search did not cover get a new search, and reset_cache clears its samples."""
        profiler = self._profiler(False)
        with _quiet():
            profiler.update_cache_iterative(params=["a", "b"], niter=1, update_1D=True, update_2D=True,
                                            num_points_1D=16, num_points_2D=8)
            density_c = profiler.get1DDensityGridData("c", num_points_1D=16)
            density_ac = profiler.get2DDensityGridData("a", "c", num_points_2D=8)
        spacing = density_c.x[1] - density_c.x[0]
        self.assertLessEqual(abs(density_c.x[np.argmax(density_c.P)] - self.mean[2]), spacing)
        self.assertTrue(np.all(np.isfinite(density_ac.P)))
        profiler.reset_cache()
        for name in ["_1d_bins", "_1d_samples", "_1d_logP", "_2d_bins", "_2d_samples", "_2d_logP"]:
            self.assertFalse(hasattr(profiler, name), name)
        with _quiet():
            profiler.update_cache(params=["b", "c"], update_MAP=False, num_points_1D=16, num_points_2D=8)
        self.assertEqual(sorted(profiler.profile_density_1D.keys()), [1, 2])
        self.assertIn(2, profiler.profile_density_2D[1])

    def test_pickle_round_trip(self):
        """savePickle stores the profiler with its flow and loadPickle restores both."""
        profiler = self._profiler(False)
        with _quiet():
            profiler.find_MAP(num_best_to_follow=5)
        profiler.get1DDensityGridData("a", num_points_1D=16)
        points = self.mean[None, :] + np.array([[0.0, 0.0, 0.0], [0.3, 0.1, -0.2]])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profiler.pt")
            profiler.savePickle(path)
            loaded = fp.posterior_profile_plotter.loadPickle(path)
            other_path = os.path.join(directory, "other.pt")
            torch.save({"not": "a profiler"}, other_path)
            with self.assertRaises(TypeError):
                fp.posterior_profile_plotter.loadPickle(other_path)
        self.assertIsInstance(loaded, fp.posterior_profile_plotter)
        self.assertTrue(np.allclose(loaded.flow_MAP, profiler.flow_MAP))
        self.assertEqual(loaded.flow_MAP_logP, profiler.flow_MAP_logP)
        self.assertEqual(list(loaded.profile_density_1D.keys()), [0])
        self.assertTrue(np.allclose(loaded.profile_density_1D[0].P, profiler.profile_density_1D[0].P))
        self.assertTrue(np.allclose(tu.to_numpy(loaded.flow.log_probability(points)),
                                    tu.to_numpy(self.flow.log_probability(points))))
        self.assertIsNotNone(loaded.getBestFit())


class TestAverageFlowProfileVariance(unittest.TestCase):
    """get_1d_profile_variance on average flows of Gaussian flows."""

    @classmethod
    def setUpClass(cls):
        """Build Gaussian flows on different chains."""
        cls.flows = []
        for seed in (1, 2):
            _, flow = _gaussian_flow(seed)
            flow.log["val_loss"] = [1.0]
            cls.flows.append(flow)

    def _average_profiler(self, flows):
        """Seeded profiler of the average of ``flows``."""
        with _quiet():
            average = sp.average_flow(flows, validation_training_idx=(flows[0].test_idx, flows[0].training_idx))
        torch.manual_seed(0)
        return fp.posterior_profile_plotter(average, feedback=0, use_scipy=False, num_minimization_samples=2000)

    def test_profile_variance(self):
        """Different flows give a positive spread around the average profile."""
        profiler = self._average_profiler(self.flows)
        x, profile, std = profiler.get_1d_profile_variance("a", normalize_by="max", num_points_1D=16)
        self.assertEqual(x.shape, (17,))
        self.assertEqual(profile.shape, x.shape)
        self.assertEqual(std.shape, x.shape)
        self.assertAlmostEqual(np.max(profile), 1.0)
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std >= 0.0))
        self.assertGreater(np.max(std), 0.0)
        with self.assertRaises(ValueError):
            profiler.get_1d_profile_variance("a", normalize_by="unknown")

    def test_profile_variance_by_index(self):
        """The parameter can be given by index, and an unknown name raises."""
        profiler = self._average_profiler(self.flows)
        x_name, profile_name, std_name = profiler.get_1d_profile_variance("a", normalize_by="max", num_points_1D=16)
        x_idx, profile_idx, std_idx = profiler.get_1d_profile_variance(0, normalize_by="max", num_points_1D=16)
        self.assertTrue(np.allclose(x_name, x_idx))
        self.assertTrue(np.allclose(profile_name, profile_idx))
        self.assertTrue(np.allclose(std_name, std_idx))
        with self.assertRaises(ValueError):
            profiler.get_1d_profile_variance("missing")

    def test_profile_variance_identical_flows(self):
        """Identical flows give no spread."""
        profiler = self._average_profiler([self.flows[0], self.flows[0]])
        _, _, std = profiler.get_1d_profile_variance("a", normalize_by="max", num_points_1D=16)
        self.assertTrue(np.allclose(std, 0.0, atol=1e-6))

    def test_profile_variance_integral(self):
        """Integral normalization gives unit area profiles."""
        profiler = self._average_profiler(self.flows)
        x, profile, std = profiler.get_1d_profile_variance("a", normalize_by="integral", num_points_1D=16)
        area = np.sum(0.5 * (profile[1:] + profile[:-1]) * np.diff(x))
        self.assertAlmostEqual(area, 1.0, places=5)
        self.assertTrue(np.all(np.isfinite(std)))

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
