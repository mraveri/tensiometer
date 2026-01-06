"""Tests for flow_profiler utilities with lightweight stubs."""

#########################################################################################################
# Imports

import os
import types
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

#########################################################################################################
# Test configuration

os.environ["NUMBA_DISABLE_JIT"] = "1"

# Import after disabling JIT so numba honors the environment flag.
import tensiometer.synthetic_probability.flow_profiler as fp

tfb = tfp.bijectors

#########################################################################################################
# Helper flows


class TinyFlow:
    """Minimal flow stub for profiler."""

    def __init__(self):
        """Init."""
        self.num_params = 2
        self.param_names = ["p0", "p1"]
        self.param_labels = ["P0", "P1"]
        self.parameter_ranges = {"p0": (-1.0, 1.0), "p1": (-1.0, 1.0)}
        self.name_tag = "tiny"
        self.chain_samples = np.zeros((3, 2), dtype=np.float32)
        self.chain_loglikes = np.zeros(3, dtype=np.float32)

    def cast(self, arr):
        """Cast."""
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    def sample(self, n):
        """Sample."""
        return tf.zeros((n, self.num_params), dtype=tf.float32)

    def log_probability(self, x):
        """Log probability."""
        return tf.zeros((tf.shape(x)[0],), dtype=tf.float32)

    def log_probability_jacobian(self, x):
        """Log probability jacobian."""
        return tf.zeros_like(x)

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return x

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return x


#########################################################################################################
# Unit tests


class TestFlowProfilerUnit(unittest.TestCase):
    """Flow profiler unit test suite."""
    def _profiler(self, **kwargs):
        """Build a profiler instance for unit tests."""
        return fp.posterior_profile_plotter(TinyFlow(), initialize_cache=False, feedback=0, **kwargs)

    def test_tiny_flow_methods(self):
        """Test TinyFlow helper methods."""
        flow = TinyFlow()
        arr = tf.zeros((2, flow.num_params), dtype=tf.float32)
        self.assertEqual(flow.log_probability_jacobian(arr).shape, arr.shape)
        self.assertTrue(np.allclose(flow.map_to_abstract_coord(arr).numpy(), arr.numpy()))
        self.assertTrue(np.allclose(flow.map_to_original_coord(arr).numpy(), arr.numpy()))

    def test_binned_argmax_utilities(self):
        """Test binned argmax utilities."""
        bins = np.array([0, 1, 0, 1])
        vals = np.array([0.1, 0.2, 0.3, 0.4])
        # call numba-wrapped functions via py_func to avoid JIT overhead in tests
        res1 = fp._binned_argmax_1D.py_func(bins, vals, 2)
        self.assertEqual(res1[0], 2)
        self.assertEqual(res1[1], 3)

        xb = np.array([0, 1])
        yb = np.array([1, 0])
        vals2 = np.array([0.5, 0.6])
        res2 = fp._binned_argmax_2D.py_func(xb, yb, vals2, 2, 2)
        self.assertEqual(res2[0, 1], 0)
        self.assertEqual(res2[1, 0], 1)

    def test_points_minimizer_tf_branch(self):
        """Test points_minimizer TensorFlow branch."""
        fake_res = types.SimpleNamespace(converged=True, objective_value=0.0, position=tf.zeros((2,)))
        with patch("tensiometer.synthetic_probability.flow_profiler.tfp.optimizer.lbfgs_minimize", return_value=fake_res):
            success, val, pt = fp.points_minimizer(lambda x: tf.constant(0.0), lambda x: x, np.array([0.0, 0.0]),
                                                   bounds=None, use_scipy=False)
        self.assertTrue(success)
        self.assertEqual(val, 0.0)
        self.assertTrue(np.allclose(pt, np.zeros(2)))

    def test_points_minimizer_scipy_branch(self):
        """Test points_minimizer SciPy branch."""
        fake_res = types.SimpleNamespace(success=True, fun=1.0, x=np.array([1.0, -1.0]), nfev=1, njev=1, message="ok")
        with patch("tensiometer.synthetic_probability.flow_profiler.minimize", return_value=fake_res) as mock_min:
            success, val, pt = fp.points_minimizer(lambda x: np.sum(x**2), lambda x: 2 * x,
                                                   [np.array([0.5, -0.5])], bounds=[(-1, 1), (-1, 1)],
                                                   use_scipy=True, feedback=2, use_jac=False)
        self.assertTrue(success[0])
        mock_min.assert_called()

    def test_par_and_number_missing(self):
        """Test _parAndNumber missing parameter handling."""
        prof = self._profiler()
        idx, name = prof._parAndNumber("missing")
        self.assertIsNone(idx)
        self.assertIsNone(name)

    def test_bestfit_and_likestats_errors(self):
        """Test best-fit and likestats error handling."""
        prof = self._profiler()
        with self.assertRaises(ValueError):
            prof.getBestFit()
        with self.assertRaises(ValueError):
            prof._initialize_likestats()

    def test_masked_gradient_ascent_runs(self):
        """Test masked gradient ascent runs."""
        prof = self._profiler()
        prof.temp_inv_cov = tf.eye(2, dtype=tf.float32)
        ensemble = tf.zeros((1, 2), dtype=tf.float32)
        mask = tf.constant([1.0, 0.0], dtype=tf.float32)
        out = prof._masked_gradient_ascent(learning_rate=0.1, num_iterations=0, ensemble=ensemble, mask=mask)
        self.assertEqual(out[0].shape[1], 2)

    def test_sample_profile_population_sets_temp(self):
        """Test sample profile population caching."""
        prof = self._profiler()
        with patch("tensiometer.synthetic_probability.flow_profiler.tf.linalg.inv",
                   return_value=tf.eye(2, dtype=tf.float32)):
            prof.sample_profile_population(num_minimization_samples=3)
        self.assertEqual(prof.temp_samples.shape[0], 3)
        self.assertEqual(prof.temp_probs.shape[0], 3)

    def test_find_flow_map_box_prior_path(self):
        """Test MAP search with a box prior."""
        prof = self._profiler(box_prior=True, use_scipy=False)
        prof.temp_samples = tf.zeros((3, 2))
        prof.temp_probs = tf.zeros((3,))
        prof.options["num_minimization_samples"] = 5
        fake_min = (np.array([True, True]), np.array([0.0, 0.0]), np.zeros((2, 2)))
        with patch.object(fp.posterior_profile_plotter, "_get_masked_box_bijector", return_value=tfb.Identity()), \
                patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer", return_value=fake_min), \
                patch("tensiometer.synthetic_probability.flow_profiler.np.argpartition",
                      return_value=np.array([0, 1, 2])):
            val, sol = prof.find_MAP()
        self.assertEqual(sol.shape, (2,))

    def test_update_cache_no_updates(self):
        """Test update_cache no-op path."""
        prof = self._profiler()
        prof.update_cache(update_MAP=False, update_1D=False, update_2D=False)
        self.assertIsNone(prof.temp_samples)
        self.assertIsNone(prof.temp_probs)

    def test_update_cache_iterative_no_updates(self):
        """Test update_cache_iterative no-op path."""
        prof = self._profiler()
        prof.update_cache_iterative(update_1D=False, update_2D=False, niter=0)
        self.assertIsNone(prof.temp_samples)
        self.assertIsNone(prof.temp_probs)

#########################################################################################################
# Additional helper flows


class SmallFlow:
    """Small Flow test suite."""
    def __init__(self, with_ranges=True):
        """Init."""
        self.num_params = 2
        self.param_names = ["p0", "p1"]
        self.param_labels = ["P0", "P1"]
        self.parameter_ranges = {"p0": (0.0, 1.0), "p1": (0.0, 1.0)} if with_ranges else None
        self.name_tag = "sf"
        self.chain_samples = np.zeros((2, 2), dtype=np.float32)
        self.chain_loglikes = np.zeros(2, dtype=np.float32)

    def cast(self, arr):
        """Cast."""
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    def sample(self, n):
        """Sample a simple grid."""
        return tf.constant(np.linspace(0, 1, n * self.num_params).reshape(n, self.num_params), dtype=tf.float32)

    def log_probability(self, x):
        """Log probability."""
        return tf.zeros((x.shape[0],), dtype=tf.float32)

    def log_probability_abs(self, x):
        """Log probability abs."""
        return tf.zeros((x.shape[0],), dtype=tf.float32)

    def log_probability_jacobian(self, x):
        """Log probability jacobian."""
        return tf.zeros((x.shape[0], self.num_params), dtype=tf.float32)

    def log_probability_abs_jacobian(self, x):
        """Log probability abs jacobian."""
        return tf.zeros((x.shape[0], self.num_params), dtype=tf.float32)

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return x

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return x


class TestFlowProfilerMore(unittest.TestCase):
    """Flow profiler additional test suite."""
    def test_small_flow_methods(self):
        """Test SmallFlow helper methods."""
        flow = SmallFlow(with_ranges=False)
        arr = tf.zeros((2, flow.num_params), dtype=tf.float32)
        self.assertEqual(flow.log_probability_abs(arr).shape, (2,))
        self.assertEqual(flow.log_probability_jacobian(arr).shape, (2, flow.num_params))
        self.assertEqual(flow.log_probability_abs_jacobian(arr).shape, (2, flow.num_params))
        self.assertTrue(np.allclose(flow.map_to_abstract_coord(arr).numpy(), arr.numpy()))
        self.assertTrue(np.allclose(flow.map_to_original_coord(arr).numpy(), arr.numpy()))

    def test_sample_profile_population_box_prior(self):
        """Test sample profile population with a box prior."""
        flow = SmallFlow(with_ranges=True)
        profiler = fp.posterior_profile_plotter(flow, initialize_cache=False, feedback=0, box_prior=True,
                                                use_scipy=False, pre_polish=False, polish=False, smoothing=False)
        with patch.object(profiler, "_get_masked_box_bijector", return_value=tfp.bijectors.Identity()), \
                patch("tensiometer.synthetic_probability.flow_profiler.tf.linalg.inv",
                      return_value=tf.eye(flow.num_params, dtype=tf.float32)):
            profiler.sample_profile_population(num_minimization_samples=4)
        self.assertIsNotNone(profiler.temp_cov)
        self.assertIsNotNone(profiler.temp_inv_cov)

    def test_find_map_scipy_path_randomized(self):
        """Test randomized MAP search with SciPy."""
        flow = SmallFlow(with_ranges=False)
        profiler = fp.posterior_profile_plotter(flow, initialize_cache=False, feedback=0, use_scipy=True,
                                                pre_polish=False, polish=False, smoothing=False)
        profiler.temp_samples = tf.constant([[0.0, 0.0], [0.5, 0.5]], dtype=tf.float32)
        profiler.temp_probs = tf.constant([0.1, 0.2], dtype=tf.float32)
        fake_pm = (np.array([True]), np.array([0.0]), np.array([[0.1, 0.2]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer", return_value=fake_pm):
            val, sol = profiler.find_MAP(randomize=True, num_best_to_follow=1)
        self.assertEqual(sol.shape[0], 2)

    def test_masked_box_bijector_and_reset(self):
        """Test masked box bijector and cache reset."""
        flow = SmallFlow(with_ranges=True)
        profiler = fp.posterior_profile_plotter(flow, initialize_cache=False, feedback=0, use_scipy=False,
                                                pre_polish=False, polish=False, smoothing=False)
        bij = profiler._get_masked_box_bijector(mask=[1, 0])
        self.assertIsInstance(bij, tfp.bijectors.Bijector)
        profiler.profile_density_1D["p0"] = "filled"
        profiler.reset_cache()
        self.assertEqual(profiler.profile_density_1D, {})
        _ = fp._binned_argmax_1D.py_func(np.array([0, 1]), np.array([1.0, 2.0]), 3)
        _ = fp._binned_argmax_2D.py_func(np.array([0, 1]), np.array([0, 1]), np.array([1.0, 2.0]), 2, 2)

#########################################################################################################
# Additional helper flows


class DummyFlowAdditional:
    """Minimal flow stub to satisfy flow_profiler interfaces."""

    def __init__(self):
        """Init."""
        self.num_params = 2
        self.param_names = ["p0", "p1"]
        self.param_labels = ["P0", "P1"]
        self.parameter_ranges = {"p0": (-1.0, 1.0), "p1": (-2.0, 2.0)}
        self.name_tag = "dummy"
        self.chain_samples = np.zeros((2, 2), dtype=np.float32)
        self.chain_loglikes = np.zeros(2, dtype=np.float32)

    def cast(self, arr):
        """Cast."""
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return x

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return x

    def sample(self, n):
        """Sample."""
        return tf.zeros((n, self.num_params), dtype=tf.float32)

    def log_probability(self, x):
        """Log probability."""
        return tf.zeros((x.shape[0],), dtype=tf.float32)

    def log_probability_abs(self, x):
        """Log probability abs."""
        return tf.zeros((x.shape[0],), dtype=tf.float32)

    def log_probability_jacobian(self, x):
        """Log probability jacobian."""
        return tf.zeros((x.shape[0], self.num_params), dtype=tf.float32)

    def log_probability_abs_jacobian(self, x):
        """Log probability abs jacobian."""
        return tf.zeros((x.shape[0], self.num_params), dtype=tf.float32)


class TestFlowProfilerAdditional(unittest.TestCase):
    """Flow profiler targeted test suite."""
    def setUp(self):
        """Set up test fixtures."""
        # Stub TensorFlow minimizer globally to prevent long runs.
        fake_tf_res = types.SimpleNamespace(converged=True, objective_value=0.0, position=np.array([0.0, 0.0]))
        self._tf_patcher = patch("tensiometer.synthetic_probability.flow_profiler.tfp.optimizer.lbfgs_minimize",
                                 return_value=fake_tf_res)
        self._tf_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self._tf_patcher.stop()

    def test_points_minimizer_scipy_and_tf_paths(self):
        """Exercise scipy and TensorFlow points_minimizer branches."""

        def func(x):
            """Simple quadratic objective."""
            return np.sum(x**2)

        def jac(x):
            """Gradient of the quadratic objective."""
            return 2 * x

        self.assertEqual(func(np.array([1.0, -1.0])), 2.0)
        self.assertTrue(np.allclose(jac(np.array([1.0, -1.0])), np.array([2.0, -2.0])))

        fake_res = types.SimpleNamespace(success=True, fun=1.0, x=np.array([0.0, 0.0]), nfev=1, njev=1, message="ok")
        with patch("tensiometer.synthetic_probability.flow_profiler.minimize", return_value=fake_res) as mock_min:
            success, val, pt = fp.points_minimizer(func, jac, [np.array([1.0, -1.0])],
                                                   bounds=[(-1, 1), (-1, 1)], feedback=2, use_scipy=True,
                                                   use_jac=False, method="L-BFGS-B",
                                                   options={"ftol": 1e-6, "gtol": 1e-5, "maxls": 10})
        self.assertTrue(success.all())
        mock_min.assert_called()
        success_tf, val_tf, pt_tf = fp.points_minimizer(func, jac, np.array([0.5, 0.5]),
                                                        bounds=[(-1, 1), (-1, 1)], use_scipy=False)
        self.assertTrue(success_tf)
        self.assertEqual(val_tf, 0.0)

    def test_dummy_flow_additional_methods(self):
        """Test DummyFlowAdditional helper methods."""
        flow = DummyFlowAdditional()
        arr = tf.zeros((2, flow.num_params), dtype=tf.float32)
        self.assertTrue(np.allclose(flow.map_to_abstract_coord(arr).numpy(), arr.numpy()))
        self.assertTrue(np.allclose(flow.map_to_original_coord(arr).numpy(), arr.numpy()))
        self.assertEqual(flow.sample(3).shape, (3, flow.num_params))
        self.assertEqual(flow.log_probability(arr).shape, (2,))
        self.assertEqual(flow.log_probability_abs(arr).shape, (2,))
        self.assertEqual(flow.log_probability_jacobian(arr).shape, (2, flow.num_params))
        self.assertEqual(flow.log_probability_abs_jacobian(arr).shape, (2, flow.num_params))

    def test_find_flow_MAP_branches(self):
        """Test find_flow_MAP branches."""
        flow = DummyFlowAdditional()
        with self.assertRaises(ValueError):
            fp.find_flow_MAP(flow, abstract=True, box_bijector=tfp.bijectors.Identity(), initial_points=tf.zeros((1, 2)))
        fake_res = (np.array([True, True]), np.array([0.0, -1.0]), np.array([[0.1, 0.2], [0.3, 0.4]]))
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer", return_value=fake_res):
            val, sol = fp.find_flow_MAP(flow, feedback=1, abstract=False, initial_points=tf.zeros((2, 2)),
                                        use_scipy=True)
        self.assertEqual(sol.shape[0], 2)
        with patch("tensiometer.synthetic_probability.flow_profiler.points_minimizer",
                   return_value=fake_res):
            val2, sol2 = fp.find_flow_MAP(flow, feedback=0, abstract=False,
                                          box_bijector=tfp.bijectors.Identity(),
                                          initial_points=tf.zeros((2, 2)),
                                          use_scipy=False)
        self.assertEqual(sol2.shape[0], 2)

    def test_posterior_profile_plotter_lightweight(self):
        """Test posterior_profile_plotter lightweight paths."""
        flow = DummyFlowAdditional()
        profiler = fp.posterior_profile_plotter(flow, initialize_cache=False, feedback=0, use_scipy=False,
                                                pre_polish=False, polish=False, smoothing=False)
        profiler.reset_cache()
        self.assertIsNone(profiler.temp_samples)
        profiler.update_cache(update_MAP=False, update_1D=False, update_2D=False)
        self.assertIsNone(profiler.temp_samples)

#########################################################################################################
# Extra helper flows


class DummyFlowExtra:
    """Minimal flow stub."""

    def __init__(self):
        """Init."""
        self.num_params = 2
        self.param_names = ["p0", "p1"]
        self.param_labels = ["P0", "P1"]
        self.parameter_ranges = {"p0": (-1.0, 1.0), "p1": (-1.0, 1.0)}
        self.name_tag = "dummy"
        self.chain_samples = np.zeros((3, 2), dtype=np.float32)
        self.chain_loglikes = np.zeros(3, dtype=np.float32)

    def cast(self, arr):
        """Cast."""
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    def sample(self, n):
        """Sample."""
        return tf.zeros((n, self.num_params), dtype=tf.float32)

    def log_probability(self, x):
        """Log probability."""
        return tf.zeros((x.shape[0],), dtype=tf.float32)

    def log_probability_jacobian(self, x):
        """Log probability jacobian."""
        return tf.zeros_like(x)

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return x

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return x


#########################################################################################################
# Extra tests


class TestFlowProfilerExtra(unittest.TestCase):
    """Flow profiler extra test suite."""
    def _make_profiler(self, **kwargs):
        """Build a profiler instance."""
        return fp.posterior_profile_plotter(DummyFlowExtra(), initialize_cache=False, feedback=0, **kwargs)

    def test_update_cache_cleanup_without_updates(self):
        """Test update_cache cleanup without updates."""
        profiler = self._make_profiler()
        profiler.update_cache(update_MAP=False, update_1D=False, update_2D=False)
        self.assertIsNone(profiler.temp_samples)
        self.assertIsNone(profiler.temp_probs)

    def test_reset_cache_clears_temp(self):
        """Test reset_cache clears cached state."""
        profiler = self._make_profiler()
        profiler.temp_samples = tf.ones((1, 2))
        profiler.temp_probs = tf.ones((1,))
        profiler.profile_density_1D = {"x": 1}
        profiler.reset_cache()
        self.assertIsNone(profiler.temp_samples)
        self.assertFalse(profiler.profile_density_1D)

    def test_dummy_flow_extra_methods(self):
        """Test DummyFlowExtra helper methods."""
        flow = DummyFlowExtra()
        arr = tf.zeros((2, flow.num_params), dtype=tf.float32)
        self.assertEqual(flow.cast(arr).shape, arr.shape)
        self.assertEqual(flow.sample(3).shape, (3, flow.num_params))
        self.assertEqual(flow.log_probability(arr).shape, (2,))
        self.assertEqual(flow.log_probability_jacobian(arr).shape, arr.shape)
        self.assertTrue(np.allclose(flow.map_to_abstract_coord(arr).numpy(), arr.numpy()))
        self.assertTrue(np.allclose(flow.map_to_original_coord(arr).numpy(), arr.numpy()))

    def test_stub_profiler_sampling(self):
        """Test StubProfiler sample_profile_population."""
        profiler = StubProfiler(DummyFlowIterative(with_ranges=True))
        profiler.sample_profile_population()
        self.assertEqual(profiler.temp_samples.shape, (2, 2))

    def test_update_cache_iterative_sets_bins(self):
        """Test update_cache_iterative sets bins."""
        profiler = self._make_profiler()
        profiler.index = {"p0": 0, "p1": 1}
        profiler._parAndNumber = lambda name: (0, types.SimpleNamespace(name="p0", label="p0"))

        def stub_sampler(self=None, **kwargs):
            """Stub sampler."""
            profiler.temp_samples = tf.zeros((2, 2), dtype=tf.float32)
            profiler.temp_probs = tf.zeros((2,), dtype=tf.float32)
            profiler.temp_inv_cov = tf.eye(2, dtype=tf.float32)

        stub_sampler()
        with patch.object(fp.posterior_profile_plotter, "sample_profile_population", new=stub_sampler):
            profiler.update_cache_iterative(params=["p0"], niter=0, update_1D=False, update_2D=False)
        self.assertIsNone(profiler.temp_samples)

    def test_initialize_likestats_profile_true(self):
        """Test _initialize_likestats with profile limits."""
        profiler = self._make_profiler()
        profiler.flow_MAP_logP = 0.0
        profiler.flow_MAP = np.zeros(2)
        profiler.temp_samples = tf.zeros((2, 2))
        profiler.temp_probs = tf.zeros((2,))
        density = types.SimpleNamespace(x=np.linspace(-1, 1, 3), P=np.ones(3))
        with patch.object(fp.mcsamples.MCSamples, "get1DDensityGridData", return_value=density):
            stats = profiler._initialize_likestats(profile_lims=True)
        self.assertIsNotNone(stats)
        self.assertTrue(hasattr(profiler, "likeStats"))

    def test_initialize_likestats_profile_false(self):
        """Test _initialize_likestats without profile limits."""
        profiler = self._make_profiler()
        profiler.flow_MAP_logP = 0.0
        profiler.flow_MAP = np.zeros(2)
        profiler.temp_samples = tf.zeros((3, 2))
        profiler.temp_probs = tf.zeros((3,))
        density = types.SimpleNamespace(x=np.linspace(-1, 1, 3), P=np.ones(3))
        with patch.object(fp.mcsamples.MCSamples, "get1DDensityGridData", return_value=density):
            stats = profiler._initialize_likestats(profile_lims=False)
        self.assertIsNotNone(stats)
        self.assertTrue(hasattr(stats, "ND_contours"))

    def test_getbestfit_raises_without_init(self):
        """Test getBestFit raises without initialization."""
        profiler = self._make_profiler()
        with self.assertRaises(ValueError):
            profiler.getBestFit()

    def test_getbestfit_after_initialize(self):
        """Test getBestFit after initialization."""
        profiler = self._make_profiler()
        profiler.flow_MAP_logP = 0.0
        profiler.flow_MAP = np.zeros(2)
        profiler._initialize_bestfit()
        bf = profiler.getBestFit()
        self.assertEqual(len(bf.names), profiler.n)

    def test_precompute_calls_get1d(self):
        """Test precompute_1D calls get1DDensityGridData."""
        profiler = self._make_profiler()
        with patch.object(fp.posterior_profile_plotter, "get1DDensityGridData") as mock_get1d:
            profiler.precompute_1D(["p0"])
        mock_get1d.assert_called_once()

    def test_get1d_density_else_branch(self):
        """Test get1DDensityGridData else branch."""
        profiler = self._make_profiler()
        profiler.index = {"p0": 0, "p1": 1}
        profiler._parAndNumber = lambda name: (0, types.SimpleNamespace(name="p0", label="p0"))
        cached_density = fp.Density1D(np.linspace(-1, 1, 3), np.ones(3))
        profiler.profile_density_1D = {0: cached_density}
        res = profiler.get1DDensityGridData("p0", num_points_1D=4)
        self.assertIsInstance(res, fp.Density1D)

    def test_permutations_iterable_branch_and_map_to_unitcube(self):
        """Test permutations iterable branch and map_to_unitcube."""
        prof = self._make_profiler(permutations=[np.array([0, 1])], map_to_unitcube=True, transformation_type="spline")
        # trigger the map_to_unitcube branch via cached bijector calls
        prof.temp_samples = tf.zeros((1, 2))
        prof.temp_probs = tf.zeros((1,))
        prof.flow_MAP_logP = 0.0
        prof.flow_MAP = np.zeros(2)
        prof._initialize_bestfit()
        self.assertIsNotNone(prof.bestfit)

#########################################################################################################
# Iterative helper flows


class DummyFlowIterative:
    """Minimal flow stub to avoid heavy computation."""

    def __init__(self, with_ranges=True):
        """Init."""
        self.num_params = 2
        self.param_names = ["p0", "p1"]
        self.param_labels = ["P0", "P1"]
        self.parameter_ranges = {"p0": (0.0, 1.0), "p1": (0.0, 1.0)} if with_ranges else None
        self.name_tag = "dummy"
        self.chain_samples = np.zeros((2, 2), dtype=np.float32)
        self.chain_loglikes = np.zeros(2, dtype=np.float32)

    def cast(self, arr):
        """Cast."""
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    def sample(self, n):
        """Sample."""
        return tf.zeros((n, self.num_params), dtype=tf.float32)

    def log_probability(self, x):
        """Log probability."""
        return tf.zeros((x.shape[0],), dtype=tf.float32)

    def log_probability_abs(self, x):
        """Log probability abs."""
        return tf.zeros((x.shape[0],), dtype=tf.float32)

    def log_probability_jacobian(self, x):
        """Log probability jacobian."""
        return tf.zeros((x.shape[0], self.num_params), dtype=tf.float32)

    def log_probability_abs_jacobian(self, x):
        """Log probability abs jacobian."""
        return tf.zeros((x.shape[0], self.num_params), dtype=tf.float32)

    def map_to_abstract_coord(self, x):
        """Map to abstract coord."""
        return x

    def map_to_original_coord(self, x):
        """Map to original coord."""
        return x


class StubProfiler(fp.posterior_profile_plotter):
    """Profiler subclass with stubbed sampling and densities."""

    def __init__(self, flow):
        """Init."""
        super().__init__(flow, initialize_cache=False, feedback=0, use_scipy=False,
                         pre_polish=False, polish=False, smoothing=False)

    def sample_profile_population(self, **kwargs):
        """Sample profile population."""
        self.temp_samples = tf.constant([[0.2, 0.3], [0.6, 0.7]], dtype=tf.float32)
        self.temp_probs = tf.constant([0.1, 0.2], dtype=tf.float32)
        self.temp_cov = tf.eye(self.n, dtype=tf.float32)
        self.temp_inv_cov = tf.eye(self.n, dtype=tf.float32)
        return None

    def get1DDensityGridData(self, name_or_idx, **kwargs):
        """Get1 D Density Grid Data."""
        return types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]))

    def get2DDensityGridData(self, ind1, ind2, **kwargs):
        """Get2 D Density Grid Data."""
        return types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]),
                                     y=np.array([0.0, 0.5, 1.0]))


#########################################################################################################
# Iterative tests


class TestFlowProfilerIterative(unittest.TestCase):
    """Flow profiler iterative test suite."""
    def setUp(self):
        """Set up test fixtures."""
        # Stub minimizers globally for safety.
        self._points_patch = patch(
            "tensiometer.synthetic_probability.flow_profiler.points_minimizer",
            return_value=(np.array([True]), np.array([0.0]), np.array([[0.0, 0.0]])),
        )
        self._points_patch.start()
        self._tf_patch = patch(
            "tensiometer.synthetic_probability.flow_profiler.tfp.optimizer.lbfgs_minimize",
            return_value=types.SimpleNamespace(converged=True, objective_value=0.0, position=np.array([0.0, 0.0])),
        )
        self._tf_patch.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self._tf_patch.stop()
        self._points_patch.stop()

    def test_binned_argmax_helpers(self):
        """Test binned argmax helpers."""
        vals = np.array([1.0, 2.0, 3.0])
        bins = np.array([0, 1, 1])
        res1 = fp._binned_argmax_1D.py_func(bins, vals, 3)
        self.assertTrue(res1[1] >= 0)
        xb = np.array([0, 1, 0])
        yb = np.array([0, 1, 1])
        res2 = fp._binned_argmax_2D.py_func(xb, yb, vals, 2, 2)
        self.assertTrue(np.any(res2 >= 0))

    def test_update_cache_iterative(self):
        """Test update_cache_iterative."""
        profiler = StubProfiler(DummyFlowIterative())
        dummy1d = types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]))
        dummy2d = types.SimpleNamespace(x=np.array([0.0, 0.5, 1.0]), y=np.array([0.0, 0.5, 1.0]))
        with patch("getdist.mcsamples.MCSamples.get1DDensityGridData", return_value=dummy1d), \
                patch("getdist.mcsamples.MCSamples.get2DDensityGridData", return_value=dummy2d), \
                patch.object(fp.posterior_profile_plotter, "_parAndNumber", side_effect=lambda name: (0, name)):
            profiler.update_cache_iterative(params=["p0", "p1"], niter=0, update_1D=True, update_2D=True)
        self.assertIn("p0", profiler._1d_samples)
        self.assertIn(("p0", "p1"), profiler._2d_samples)
        profiler.reset_cache()
        self.assertEqual(profiler.profile_density_1D, {})

    def test_find_map_and_box_bijector(self):
        """Test find_MAP and box bijector branches."""
        flow = DummyFlowIterative(with_ranges=False)
        profiler = fp.posterior_profile_plotter(flow, initialize_cache=False, feedback=0, use_scipy=False,
                                                pre_polish=False, polish=False, smoothing=False)
        val, sol = profiler.find_MAP(x0=tf.zeros((1, flow.num_params)), randomize=False, abstract=True)
        self.assertEqual(sol.shape[0], flow.num_params)
        flow_with_ranges = DummyFlowIterative(with_ranges=True)
        profiler2 = fp.posterior_profile_plotter(flow_with_ranges, initialize_cache=False, feedback=0, use_scipy=False,
                                                 pre_polish=False, polish=False, smoothing=False, box_prior=True)
        bij = profiler2._get_masked_box_bijector(mask=[1, 0])
        self.assertIsInstance(bij, tfp.bijectors.Bijector)

    def test_dummy_flow_iterative_methods(self):
        """Test DummyFlowIterative helper methods."""
        flow = DummyFlowIterative(with_ranges=True)
        arr = tf.zeros((2, flow.num_params), dtype=tf.float32)
        self.assertEqual(flow.sample(2).shape, (2, flow.num_params))
        self.assertEqual(flow.log_probability(arr).shape, (2,))
        self.assertEqual(flow.log_probability_abs(arr).shape, (2,))
        self.assertEqual(flow.log_probability_jacobian(arr).shape, (2, flow.num_params))
        self.assertEqual(flow.log_probability_abs_jacobian(arr).shape, (2, flow.num_params))
        self.assertTrue(np.allclose(flow.map_to_abstract_coord(arr).numpy(), arr.numpy()))
        self.assertTrue(np.allclose(flow.map_to_original_coord(arr).numpy(), arr.numpy()))

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
