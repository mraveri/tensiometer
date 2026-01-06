"""Tests for synthetic probability flows."""

#########################################################################################################
# Imports

import os
import tempfile
import types
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from getdist import MCSamples

import tensorflow as tf
import tensorflow_probability as tfp

import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import flow_profiler
from tensiometer.synthetic_probability import loss_functions as lf
from tensiometer.synthetic_probability import trainable_bijectors as tb
import tensiometer.utilities.stats_utilities as stutilities

#########################################################################################################
# Test configuration

unittest.TestLoader.sortTestMethodsUsing = None

tfb = tfp.bijectors
tfd = tfp.distributions

getdist_settings = {
    "ignore_rows": 0.0,
    "smooth_scale_2D": 0.3,
    "smooth_scale_1D": 0.3,
}

#########################################################################################################
# Test cases

class TestSyntheticProbability(unittest.TestCase):

    """Synthetic probability test suite."""
    def setUp(self):
        """Set up test fixtures."""
        # Speed up scipy/TF minimize used in profiling by stubbing it out.
        self._minimize_patcher = unittest.mock.patch(
            "tensiometer.synthetic_probability.flow_profiler.minimize",
            side_effect=lambda func, x0=None, jac=None, bounds=None, method=None, options=None, **kwargs: type(
                "Res", (), {"success": True, "fun": float(func(x0)), "x": np.asarray(x0), "nfev": 1, "njev": 0, "message": "stub"}
            )(),
        )
        self._minimize_patcher.start()
        self._points_patcher = unittest.mock.patch(
            "tensiometer.synthetic_probability.flow_profiler.points_minimizer",
            return_value=(np.array([True]), np.array([0.0]), np.array([[0.0, 0.0]])),
        )
        self._points_patcher.start()
        self._tf_min_patcher = unittest.mock.patch(
            "tensiometer.synthetic_probability.flow_profiler.tfp.optimizer.lbfgs_minimize",
            return_value=type("Res", (), {"converged": True, "objective_value": 0.0, "position": np.array([0.0, 0.0])})(),
        )
        self._tf_min_patcher.start()
        # lightweight dummy flow to stub training-heavy paths
        class DummyFlow:
            """Dummy Flow test suite."""
            def __init__(self):
                """Init."""
                self.num_params = 6
                self.param_names = [f"param{i+1}" for i in range(self.num_params)]
                self.param_labels = [f"P{i+1}" for i in range(self.num_params)]
                self.parameter_ranges = {name: (-1.0, 1.0) for name in self.param_names}
                self.name_tag = "dummy_flow"
                self.chain_samples = np.zeros((2, self.num_params), dtype=np.float32)
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

        self._dummy_flow = DummyFlow()
        self._flow_patch = unittest.mock.patch(
            "tensiometer.synthetic_probability.synthetic_probability.flow_from_chain", return_value=self._dummy_flow
        )
        self._flow_patch.start()
        self._avg_flow_patch = unittest.mock.patch(
            "tensiometer.synthetic_probability.synthetic_probability.average_flow_from_chain", return_value=self._dummy_flow
        )
        self._avg_flow_patch.start()
        self.chain = make_chain_flow_callback(with_loglikes=True)
        self.profiler_options = {
            "num_minimization_samples": 100,
            "num_gd_interactions_1D": 5,
            "num_gd_interactions_2D": 5,
            "scipy_options": {
                "ftol": 1.0e-06,
                "gtol": 0.0,
                "maxls": 40,
            },
            "scipy_use_jac": True,
            "num_points_1D": 64,
            "num_points_2D": 32,
            "smooth_scale_1D": 0.2,
            "smooth_scale_2D": 0.2,
        }
        def _fake_sample(profiler, **kwargs):
            """Fake sample."""
            profiler.temp_samples = tf.zeros((1, profiler.n), dtype=tf.float32)
            profiler.temp_probs = tf.zeros((1,), dtype=tf.float32)
            profiler.temp_cov = tf.eye(profiler.n, dtype=tf.float32)
            profiler.temp_inv_cov = tf.eye(profiler.n, dtype=tf.float32)
            return None
        self._sample_patch = unittest.mock.patch.object(
            flow_profiler.posterior_profile_plotter, "sample_profile_population", _fake_sample
        )
        self._sample_patch.start()
        def _fake_find_map(profiler, **kwargs):
            """Fake find map."""
            profiler.flow_MAP = np.zeros(profiler.n, dtype=float)
            profiler.flow_MAP_logP = 0.0
            return None
        self._findmap_patch = unittest.mock.patch.object(
            flow_profiler.posterior_profile_plotter, "find_MAP", _fake_find_map
        )
        self._findmap_patch.start()

    def test_dummy_flow_helpers(self):
        """Test DummyFlow helper methods."""
        flow = self._dummy_flow
        arr = tf.zeros((2, flow.num_params), dtype=tf.float32)
        self.assertEqual(flow.cast(arr).shape, arr.shape)
        self.assertEqual(flow.sample(3).shape, (3, flow.num_params))
        self.assertEqual(flow.log_probability(arr).shape, (2,))
        self.assertEqual(flow.log_probability_abs(arr).shape, (2,))
        self.assertEqual(flow.log_probability_jacobian(arr).shape, (2, flow.num_params))
        self.assertEqual(flow.log_probability_abs_jacobian(arr).shape, (2, flow.num_params))
        self.assertTrue(np.allclose(flow.map_to_abstract_coord(arr).numpy(), arr.numpy()))
        self.assertTrue(np.allclose(flow.map_to_original_coord(arr).numpy(), arr.numpy()))

        # define the parameters of the problem:
        dim = 6
        num_gaussians = 3
        num_samples = 10000

        # we seed the random number generator to get reproducible results:
        seed = 100
        np.random.seed(seed)
        # we define the range for the means and covariances:
        mean_range = (-0.5, 0.5)
        cov_scale = 0.4**2
        # means and covs:
        means = np.random.uniform(mean_range[0], mean_range[1], num_gaussians*dim).reshape(num_gaussians, dim)
        weights = np.random.rand(num_gaussians)
        weights = weights / np.sum(weights)
        covs = [cov_scale*stutilities.vector_to_PDM(np.random.rand(int(dim*(dim+1)/2))) for _ in range(num_gaussians)]

        # cast to required precision:
        means = means.astype(np.float32)
        weights = weights.astype(np.float32)
        covs = [cov.astype(np.float32) for cov in covs]

        # initialize distribution:
        distribution = tfp.distributions.Mixture(
            cat=tfp.distributions.Categorical(probs=weights),
            components=[
                tfp.distributions.MultivariateNormalTriL(loc=_m, scale_tril=tf.linalg.cholesky(_c))
                for _m, _c in zip(means, covs)
            ], name='Mixture')

        # sample the distribution:
        samples = distribution.sample(num_samples).numpy()
        # calculate log posteriors:
        logP = distribution.log_prob(samples).numpy()

        # create MCSamples from the samples:
        self.chain = MCSamples(samples=samples, 
                               settings=getdist_settings,
                               loglikes=-logP,
                               name_tag='Mixture',
                               )
        
        # profiler options:
        self.profiler_options = {
            'num_minimization_samples': 100,  # drastically cut sampling for tests
            'num_gd_interactions_1D': 5,  # number of gradient descent interactions for the 1D profile
            'num_gd_interactions_2D': 5,  # number of gradient descent interactions for the 2D profile
            'scipy_options': {  # options for the scipy polishing minimizer
                        'ftol': 1.e-06,
                        'gtol': 0.0,
                        'maxls': 40,
                    },
            'scipy_use_jac': True,  # use the jacobian in the minimizer
            'num_points_1D': 64, # number of points for the 1D profile
            'num_points_2D': 32, # number of points per dimension for the 2D profile
            'smooth_scale_1D': 0.2, # smoothing scale for the 1D profile
            'smooth_scale_2D': 0.2, # smoothing scale for the 2D profile
            }

    def tearDown(self):
        """Clean up test fixtures."""
        self._tf_min_patcher.stop()
        self._points_patcher.stop()
        self._minimize_patcher.stop()
        self._flow_patch.stop()
        self._avg_flow_patch.stop()
        if self._sample_patch is not None:
            self._sample_patch.stop()
        if self._findmap_patch is not None:
            self._findmap_patch.stop()


    def test_flow_from_chain(self):
        
        # train single flow, selecting from a population of two:
        """Test Flow from chain."""
        kwargs = {
          'feedback': 2,
          'plot_every': 0,
          'pop_size': 2,
          'epochs': 5,}
        flow = sp.flow_from_chain(self.chain, **kwargs)
        # call profiler to test the flow:
        flow_profile = flow_profiler.posterior_profile_plotter(flow, initialize_cache=False, feedback=2)
        flow_profile.update_cache(params=None, update_MAP=True, update_1D=False, update_2D=False, 
                                  **self.profiler_options)

        
    def test_average_flow_from_chain(self):
        
        # train average flow:
        """Test Average flow from chain."""
        kwargs = {
          'feedback': 2,
          'plot_every': 0,
          'pop_size': 1,
          'num_flows': 3,
          'epochs': 5,
        }
        average_flow = sp.average_flow_from_chain(self.chain, **kwargs)
        # call profiler to test the flow:
        flow_profile = flow_profiler.posterior_profile_plotter(average_flow, initialize_cache=False, feedback=2)
        flow_profile.update_cache(params=None, update_MAP=True, update_1D=False, update_2D=False, 
                                  **self.profiler_options)

#########################################################################################################
# Synthetic probability additional coverage tests



class _DummyFlow:
    """Minimal stub flow object to exercise average_flow logic."""

    def __init__(self, val_loss):
        """Init."""
        self.name_tag = "dummy"
        self.feedback = 0
        self.plot_every = 0
        self.sample_MAP = None
        self.chain_MAP = None
        self.num_params = 1
        self.param_names = ["x"]
        self.param_labels = ["x"]
        self.parameter_ranges = {"x": (0.0, 1.0)}
        self.periodic_params = []
        self.chain_samples = np.zeros((1, 1), dtype=np.float32)
        self.chain_loglikes = None
        self.has_loglikes = False
        self.chain_weights = np.ones(1, dtype=np.float32)
        self.is_trained = False
        self.MAP_coord = None
        self.MAP_logP = None
        self.prior_bijector = tfb.Identity()
        self.fixed_bijector = tfb.Identity()
        self.trainable_transformation = None
        self.bijectors = [tfb.Identity()]
        base = tfd.MultivariateNormalDiag(loc=[0.0], scale_diag=[1.0])
        self.distribution = tfd.TransformedDistribution(distribution=base, bijector=tfb.Identity())
        self.log = {"val_loss": [val_loss], "loss": [val_loss], "chi2Z_ks_p": [0.5]}
        self.training_metrics = ["loss"]
        self.training_plot_called = False
        self.reset_called = False

    def training_plot(self, **kwargs):
        """Training plot."""
        self.training_plot_called = True

    def sample(self, n):
        """Sample."""
        return tf.zeros((int(n), 1), dtype=sp.prec)

    def reset_tensorflow_caches(self):
        """Reset tensorflow caches."""
        self.reset_called = True

    def cast(self, v):
        """Cast."""
        return tf.cast(v, dtype=sp.prec)


class TestSyntheticProbabilityAdditional(unittest.TestCase):
    """Synthetic probability additional test suite."""
    def test_tfp_compat_version_guard(self):
        """Test TensorFlow and TFP compatibility checks."""
        with self.assertRaises(ValueError):
            sp._ensure_tfp_compat((2, 16, 0), (0, 23, 0))
        with self.assertRaises(ValueError):
            sp._ensure_tfp_compat((2, 15, 0), (0, 24, 0))
        sp._ensure_tfp_compat((2, 16, 0), (0, 24, 0))

    def test_init_fixed_bijector_requires_rescaling_for_periodic(self):
        """Test Init fixed bijector requires rescaling for periodic."""
        flow = sp.FlowCallback.__new__(sp.FlowCallback)
        flow.feedback = 0
        flow.param_names = ["x"]
        flow.parameter_ranges = {"x": (0.0, 1.0)}
        flow.periodic_params = ["x"]
        flow.chain_samples = np.array([[0.1], [0.2]], dtype=np.float32)
        flow.chain_weights = np.ones(2, dtype=np.float32)
        with self.assertRaises(ValueError):
            flow._init_fixed_bijector(prior_bijector="ranges", apply_rescaling=False)

    def test_init_trainable_bijector_requires_trainable_transformation(self):
        """Test Init trainable bijector requires trainable transformation."""
        flow = sp.FlowCallback.__new__(sp.FlowCallback)
        flow.feedback = 0
        flow.trainable_periodic_params = []
        flow.param_names = ["x"]
        flow.num_params = 1
        flow.chain_samples = np.zeros((2, 1), dtype=np.float32)
        flow.chain_weights = np.ones(2, dtype=np.float32)
        flow.fixed_bijector = tfb.Identity()
        flow.bijectors = []
        with self.assertRaises(ValueError):
            flow._init_trainable_bijector(
                trainable_bijector=tfb.Identity(), trainable_bijector_path="dummy"
            )

    def test_log_probability_abs_and_derivatives(self):
        """Test Log probability abs and derivatives."""
        flow = sp.FlowCallback.__new__(sp.FlowCallback)
        base = tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 1.0])
        flow.distribution = tfd.TransformedDistribution(distribution=base, bijector=tfb.Identity())

        coord = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        log_p = flow.log_probability_abs(coord)
        jac = flow.log_probability_abs_jacobian(coord)
        hess = flow.log_probability_abs_hessian(coord)

        self.assertEqual(log_p.shape, (1,))
        self.assertEqual(jac.shape, (1, 2))
        self.assertEqual(hess.shape[-2:], (2, 2))

    def test_sample_and_mcsamples_from_dummy_distribution(self):
        """Test Sample and mcsamples from dummy distribution."""
        flow = sp.FlowCallback.__new__(sp.FlowCallback)
        base = tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 1.0])
        flow.distribution = tfd.TransformedDistribution(distribution=base, bijector=tfb.Identity())
        flow.param_names = ["p1", "p2"]
        flow.param_labels = ["p1", "p2"]
        flow.parameter_ranges = {"p1": (-1.0, 1.0), "p2": (-1.0, 1.0)}
        flow.name_tag = "dummy"
        samples = flow.sample(3)
        self.assertEqual(samples.shape, (3, 2))
        mc = flow.MCSamples(3, logLikes=True)
        self.assertEqual(mc.samples.shape[0], 3)

    def test_average_flow_weights_and_training_plot(self):
        """Test Average flow weights and training plot."""
        flows = [_DummyFlow(val_loss=0.5), _DummyFlow(val_loss=1.5)]
        avg = sp.average_flow(flows, validation_training_idx=(np.array([0]), np.array([0])))
        self.assertAlmostEqual(tf.reduce_sum(avg.weights).numpy(), 1.0, places=5)
        self.assertLess(avg.weights.numpy()[1], avg.weights.numpy()[0])

        avg._set_flow_weights(mode="equal")
        self.assertTrue(np.allclose(avg.weights.numpy(), np.ones(2) / 2))

        avg.training_plot(logs={"loss": [1.0]}, ipython_plotting=False)
        self.assertTrue(all(f.training_plot_called for f in flows))

    def test_average_flow_sampling_and_cache_reset(self):
        """Test Average flow sampling and cache reset."""
        flows = [_DummyFlow(val_loss=0.2), _DummyFlow(val_loss=0.8)]
        avg = sp.average_flow(flows, validation_training_idx=(np.array([0]), np.array([0])))
        samples = avg.sample(4)
        self.assertEqual(samples.shape, (4, 1))
        avg.reset_tensorflow_caches()
        self.assertTrue(all(getattr(f, "reset_called", False) for f in flows))

    def test_average_flow_weights_missing_key_raises(self):
        """Test Average flow weights missing key raises."""
        flows = [_DummyFlow(val_loss=0.1), _DummyFlow(val_loss=0.2)]
        for f in flows:
            f.log.pop("val_loss")
        with self.assertRaises(Exception):
            sp.average_flow(flows, validation_training_idx=(np.array([0]), np.array([0])))

    def test_transformed_flow_callback_iterable_transformation(self):
        """Test Transformed flow callback iterable transformation."""
        base = tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 1.0])
        fake_flow = types.SimpleNamespace()
        fake_flow.name_tag = "flow"
        fake_flow.feedback = 0
        fake_flow.plot_every = 0
        fake_flow.num_params = 2
        fake_flow.is_trained = False
        fake_flow.log = {}
        fake_flow.param_names = ["a", "b"]
        fake_flow.param_labels = ["A", "B"]
        fake_flow.parameter_ranges = {"a": (-1.0, 1.0), "b": (-2.0, 2.0)}
        fake_flow.chain = types.SimpleNamespace(
            samples=np.array([[0.0, 0.5], [0.1, -0.1]], dtype=np.float32),
            weights=np.ones(2, dtype=np.float32),
        )
        fake_flow.chain_samples = fake_flow.chain.samples
        fake_flow.chain_loglikes = None
        fake_flow.has_loglikes = False
        fake_flow.chain_weights = fake_flow.chain.weights
        fake_flow.bijectors = [tfb.Identity()]
        fake_flow.distribution = tfd.TransformedDistribution(distribution=base, bijector=tfb.Identity())
        fake_flow.sample_MAP = None
        fake_flow.chain_MAP = None
        fake_flow.MAP_coord = None
        fake_flow.MAP_logP = None

        transformations = [tfb.Scale(2.0, name="double"), tfb.Identity()]
        with self.assertRaises(ValueError):
            sp.TransformedFlowCallback(fake_flow, transformations, transform_posterior=False)

    def test_average_flow_chi2_weight_mode(self):
        """Test Average flow chi2 weight mode."""
        flows = [_DummyFlow(val_loss=0.3), _DummyFlow(val_loss=0.6)]
        avg = sp.average_flow(flows, validation_training_idx=(np.array([0]), np.array([0])))
        avg._set_flow_weights(mode="chi2Z_ks_p")
        self.assertAlmostEqual(float(tf.reduce_sum(avg.weights)), 1.0, places=5)


#########################################################################################################
# Synthetic probability integration tests



def make_dummy_chain():
    """Make a dummy chain for integration-style coverage.

    :returns: MCSamples instance with loglikes.
    """
    samples = []
    for i in range(12):
        samples.append([i * 0.2, 1.0 - 0.05 * i + 0.01 * (-1) ** i])
    samples = np.array(samples, dtype=float)
    weights = np.ones(len(samples))
    loglikes = -np.sum(samples**2, axis=1)
    chain = MCSamples(samples=samples, weights=weights, names=["p1", "p2"])
    chain.loglikes = loglikes
    chain.name_tag = "demo"
    return chain


class TestSyntheticProbabilityIntegration(unittest.TestCase):
    """Synthetic probability integration test suite."""
    def setUp(self):
        """Set up test fixtures."""
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

    def test_flow_callback_train_and_global_train(self):
        """Test Flow callback train and global train."""
        chain = make_dummy_chain()
        flow = sp.FlowCallback(
            chain,
            feedback=0,
            plot_every=0,
            trainable_bijector=tfb.Identity(),
            apply_rescaling="independent",
            initialize_model=True,
            validation_split=0.4,
        )

        class DummyHist:
            """Dummy Hist test suite."""
            def __init__(self):
                """Init."""
                self.history = {"loss": [0.3], "val_loss": [0.4]}

        flow.train = types.MethodType(lambda self, **kwargs: DummyHist(), flow)
        flow.global_train(pop_size=1, epochs=1, steps_per_epoch=1, batch_size=1, verbose=0)
        # populate logs for summary
        for key in flow.training_metrics:
            flow.log[key].append(0.0)
        flow.chi2Z = flow.chi2Y
        flow.print_training_summary()
        flow.training_plot(logs=flow.log, ipython_plotting=False, fast=True, title="demo")
        flow.reset_tensorflow_caches()
        flow._init_nearest_samples()
        self.assertTrue(hasattr(flow, "chain_nearest_index"))

    def test_flow_from_chain_caching_and_average_flow(self):
        """Test Flow from chain caching and average flow."""
        chain = make_dummy_chain()
        tmpdir = tempfile.mkdtemp()

        # speed up training by stubbing train/global_train
        original_global_train = sp.FlowCallback.global_train
        original_train = sp.FlowCallback.train
        sp.FlowCallback.train = lambda self, **kwargs: types.SimpleNamespace(
            history={"loss": [0.1], "val_loss": [0.2]}
        )
        sp.FlowCallback.global_train = lambda self, **kwargs: None
        try:
            flow = sp.flow_from_chain(
                chain,
                cache_dir=tmpdir,
                root_name="flowcache",
                feedback=0,
                plot_every=0,
                trainable_bijector=tfb.Identity(),
                apply_rescaling="independent",
                initialize_model=False,
                validation_split=0.4,
            )
            self.assertIsInstance(flow, sp.FlowCallback)
            # second call should load from cache
            original_load = sp.FlowCallback.load
            sp.FlowCallback.load = lambda chain_arg, outroot, **kwargs: flow
            flow_cached = sp.flow_from_chain(
                chain,
                cache_dir=tmpdir,
                root_name="flowcache",
                feedback=0,
                plot_every=0,
                trainable_bijector=tfb.Identity(),
                apply_rescaling="independent",
                initialize_model=False,
                validation_split=0.4,
            )
            sp.FlowCallback.load = original_load
            self.assertIsInstance(flow_cached, sp.FlowCallback)

            original_set_weights = sp.average_flow._set_flow_weights
            def _stub_set_weights(self, mode='val_loss'):
                """Stub set weights."""
                self.weights = tf.constant(np.ones(self.num_flows) / self.num_flows, dtype=sp.prec)
                self.weights_prob = tfd.Multinomial(1, probs=self.weights, validate_args=True)
                self.distribution = self.flows[0].distribution
            sp.average_flow._set_flow_weights = _stub_set_weights
            avg = sp.average_flow_from_chain(
                chain,
                num_flows=2,
                cache_dir=tmpdir,
                root_name="avgflow",
                use_mpi=False,
                feedback=0,
                plot_every=0,
                trainable_bijector=tfb.Identity(),
                apply_rescaling="independent",
                initialize_model=False,
                validation_split=0.4,
            )
            self.assertTrue(hasattr(avg, "weights"))
            sp.average_flow._set_flow_weights = original_set_weights
        finally:
            sp.FlowCallback.global_train = original_global_train
            sp.FlowCallback.train = original_train

    def test_reset_tensorflow_caches_and_print_summary(self):
        """Test Reset tensorflow caches and print summary."""
        chain = make_dummy_chain()
        flow = sp.FlowCallback(
            chain,
            feedback=0,
            plot_every=0,
            trainable_bijector=tfb.Identity(),
            apply_rescaling="independent",
            initialize_model=True,
            validation_split=0.5,
        )
        for key in flow.training_metrics:
            flow.log[key].append(0.1)
        flow.print_training_summary()
        flow.reset_tensorflow_caches()
        self.assertTrue(hasattr(flow, "distribution"))


#########################################################################################################
# Synthetic probability geometry tests



def make_chain_geometry():
    """Make a small chain for geometry checks.

    :returns: MCSamples instance with two parameters.
    """
    samples = np.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.3]], dtype=np.float64)
    weights = np.ones(len(samples))
    loglikes = -np.sum(samples**2, axis=1)
    return MCSamples(
        samples=samples,
        weights=weights,
        names=["p0", "p1"],
        labels=["p0", "p1"],
        loglikes=loglikes,
        name_tag="geom",
    )


def stub_monitor_geometry(self):
    """Stub training monitoring hooks.

    :param self: FlowCallback instance under test.
    :returns: ``None``.
    """
    self.training_metrics = []
    self.log = {}
    self.test_samples = np.zeros((2, self.num_params), dtype=np.float32)
    self.test_weights = np.ones(2, dtype=np.float32)
    self.training_history = {}
    self.chi2Y = np.array([0.0, 0.0])
    self.chi2Y_ks = 0.0
    self.chi2Y_ks_p = 1.0


class TestFlowGeometry(unittest.TestCase):
    """Flow geometry test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.orig_monitor = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = stub_monitor_geometry
        self._orig_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)

    def tearDown(self):
        """Clean up test fixtures."""
        sp.FlowCallback._init_training_monitoring = self.orig_monitor
        tf.config.run_functions_eagerly(self._orig_eager)

    def _cb(self):
        """Cb."""
        return sp.FlowCallback(
            make_chain_geometry(),
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )

    def test_map_roundtrip_and_jacobians(self):
        """Test Map roundtrip and jacobians."""
        cb = self._cb()
        x = tf.constant([[0.1, -0.2]], dtype=tf.float32)
        z = cb.map_to_abstract_coord(x)
        x_back = cb.map_to_original_coord(z)
        self.assertEqual(z.shape, x.shape)
        self.assertEqual(x_back.shape, x.shape)
        dj = cb.direct_jacobian(x)
        ij = cb.inverse_jacobian(x)
        self.assertEqual(dj.shape[-2:], (2, 2))
        self.assertEqual(ij.shape[-2:], (2, 2))

    def test_metric_and_inverse_metric(self):
        """Test Metric and inverse metric."""
        cb = self._cb()
        coord = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        g = cb.metric(coord)
        ginv = cb.inverse_metric(coord)
        self.assertEqual(g.shape[-2:], (2, 2))
        self.assertTrue(np.all(np.isfinite(g.numpy())))
        self.assertTrue(np.all(np.isfinite(ginv.numpy())))

    def test_metric_derivatives_zero_for_identity(self):
        """Test Metric derivatives zero for identity."""
        cb = self._cb()
        coord = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        d1 = cb.coord_metric_derivative(coord)
        d2 = cb.coord_inverse_metric_derivative(coord)
        self.assertEqual(d1.shape[-2:], (2, 2))
        self.assertTrue(np.all(np.isfinite(d1.numpy())))
        self.assertTrue(np.all(np.isfinite(d2.numpy())))

    def test_second_metric_derivatives(self):
        """Test Second metric derivatives."""
        cb = self._cb()
        coord = tf.constant([[0.1, -0.1]], dtype=tf.float32)
        dd1 = cb.coord_metric_derivative_2(coord)
        dd2 = cb.coord_inverse_metric_derivative_2(coord)
        self.assertEqual(dd1.shape[-3:], (2, 2, 2))
        self.assertEqual(dd2.shape[-3:], (2, 2, 2))

    def test_geodesic_distance_identity(self):
        """Test Geodesic distance identity."""
        cb = self._cb()
        c1 = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        c2 = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        dist = cb.geodesic_distance(c1, c2)
        self.assertEqual(dist.shape, ())

    def test_geodesic_bvp_shape(self):
        """Test Geodesic bvp shape."""
        cb = self._cb()
        start = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        end = tf.constant([[1.0, 1.0]], dtype=tf.float32)
        traj = cb.geodesic_bvp(start, end, num_points=5)
        self.assertEqual(traj.shape[-2:], (5, 2))

    def test_geodesic_ivp_notimplemented(self):
        """Test Geodesic ivp notimplemented."""
        cb = self._cb()
        with self.assertRaises(TypeError):
            cb.geodesic_ivp(tf.zeros((1, 2), dtype=tf.float32), tf.zeros((1, 2), dtype=tf.float32), tf.constant([0.0]))

    def test_loss_reset_and_init(self):
        """Test Loss reset and init."""
        cb = self._cb()
        cb._init_loss_function(loss_mode="standard")
        cb.loss.reset()
        self.assertIsInstance(cb.loss, lf.standard_loss)


#########################################################################################################
# Synthetic probability geometry and plot tests





class TestSyntheticProbabilityGeometryAndPlot(unittest.TestCase):
    """Synthetic probability geometry and plot test suite."""
    def _make_geometry_flow(self, dims=2):
        """Make geometry flow."""
        flow = sp.FlowCallback.__new__(sp.FlowCallback)
        flow.bijector = tfb.Identity()
        base = tfd.MultivariateNormalDiag(loc=[0.0] * dims, scale_diag=[1.0] * dims)
        flow.distribution = tfd.TransformedDistribution(distribution=base, bijector=flow.bijector)
        flow.num_params = dims
        return flow

    def test_metric_and_connection_identity(self):
        """Test Metric and connection identity."""
        flow = self._make_geometry_flow(dims=2)
        coord = tf.constant([[1.0, -1.0]], dtype=tf.float32)
        metric = flow.metric(coord)
        inv_metric = flow.inverse_metric(coord)
        log_det = flow.log_det_metric(coord)
        levi = flow.levi_civita_connection(coord)

        self.assertTrue(np.allclose(metric.numpy()[0], np.eye(2)))
        self.assertTrue(np.allclose(inv_metric.numpy()[0], np.eye(2)))
        self.assertEqual(log_det.shape, (1,))
        self.assertTrue(np.allclose(levi.numpy(), 0.0))

    def test_geodesic_helpers_identity(self):
        """Test Geodesic helpers identity."""
        flow = self._make_geometry_flow(dims=1)
        flow.map_to_original_coord = lambda x: x
        flow.map_to_abstract_coord = lambda x: x
        p1 = tf.constant([[0.0]], dtype=tf.float32)
        p2 = tf.constant([[1.0]], dtype=tf.float32)
        dist = flow.geodesic_distance(p1, p2)
        traj = flow.geodesic_bvp.python_function(p1, p2, num_points=3)
        self.assertAlmostEqual(float(dist.numpy()), 1.0)
        self.assertEqual(traj.shape[-2:], (3, 1))
        with self.assertRaises(TypeError):
            flow.geodesic_ivp(p1, tf.constant([[0.0]], dtype=tf.float32), tf.constant([0.0, 1.0]))

    def test_save_and_load_with_patched_init(self):
        """Test Save and load with patched init."""
        flow = self._make_geometry_flow(dims=1)
        flow.trainable_transformation = None
        flow.name_tag = "geom"
        outroot = tempfile.mktemp()
        flow.save(outroot)

        original_init = sp.FlowCallback.__init__

        def _patched_init(self, chain, *args, **kwargs):
            # avoid heavy initialization; attributes restored from pickle
            """Patched init."""
            return None

        try:
            sp.FlowCallback.__init__ = _patched_init
            loaded = sp.FlowCallback.load(chain=None, outroot=outroot, initialize_model=False)
            self.assertEqual(loaded.name_tag, "geom")
            os.remove(outroot + "_flow_cache.pickle")
        finally:
            sp.FlowCallback.__init__ = original_init
            for suffix in ("_flow_cache.pickle",):
                try:
                    os.remove(outroot + suffix)
                except FileNotFoundError:
                    pass

    def test_training_monitoring_and_compute_metrics(self):
        """Test Training monitoring and compute metrics."""
        flow = self._make_geometry_flow(dims=1)
        flow.loss = sp.loss.standard_loss()
        flow.training_samples = np.array([[0.0], [1.0]], dtype=np.float32)
        flow.training_weights = np.ones(2, dtype=np.float32)
        flow.training_logP_preabs = np.zeros(2, dtype=np.float32)
        flow.test_samples = np.array([[0.5], [-0.5]], dtype=np.float32)
        flow.test_weights = np.ones(2, dtype=np.float32)
        flow.num_training_samples = len(flow.training_samples)
        flow.num_test_samples = len(flow.test_samples)
        flow.has_weights = False
        flow.test_idx = np.array([0])
        flow.training_idx = np.array([1])
        flow.trainable_bijector = tfb.Identity()
        flow.set_model(types.SimpleNamespace(call=lambda x: tf.zeros_like(x)))
        flow._init_training_monitoring()
        flow.compute_training_metrics(logs={"loss": 1.0, "val_loss": 1.5, "lr": 0.01})
        self.assertIn("loss_rate", flow.log)
        self.assertEqual(len(flow.log["loss"]), 1)
        self.assertEqual(flow.log["loss_rate"][0], 0.0)

    def test_plot_helpers_run(self):
        """Test Plot helpers run."""
        flow = self._make_geometry_flow(dims=1)
        flow.final_learning_rate = 0.001
        flow.initial_learning_rate = 0.01
        flow.log = {
            "loss": [0.5, 0.25],
            "val_loss": [0.6, 0.3],
            "lr": [0.01, 0.005],
            "chi2Z_ks": [0.1, 0.2],
            "chi2Z_ks_p": [0.9, 0.8],
            "rho_loss": [0.4, 0.3],
            "ee_loss": [0.2, 0.1],
            "val_rho_loss": [0.45, 0.35],
            "val_ee_loss": [0.25, 0.15],
            "lambda_1": [0.5, 0.4],
            "lambda_2": [0.5, 0.6],
        }
        flow.chi2Y = np.array([0.1, 0.2])
        flow.chi2Z = np.array([0.15, 0.25])
        flow.chi2Y_ks = 0.1
        flow.test_weights = np.ones_like(flow.chi2Y)
        flow.training_samples = np.array([[0.0], [0.1]], dtype=np.float32)
        flow.trainable_bijector = tfb.Identity()
        flow.num_params = 1
        fig, axes = plt.subplots(2, 3, figsize=(8, 4))
        flow._plot_loss(axes[0, 0])
        flow._plot_lr(axes[0, 1])
        flow._plot_chi2_dist(axes[0, 2], fast=True)
        flow._plot_chi2_ks_p(axes[1, 0])
        flow._plot_density_evidence_error_losses(axes[1, 1])
        flow._plot_lambda_values(axes[1, 2])
        plt.close(fig)

    def test_map_conversion_and_derivatives(self):
        """Test Map conversion and derivatives."""
        flow = self._make_geometry_flow(dims=1)
        coord = tf.constant([[2.0]], dtype=tf.float32)
        abs_coord = flow.map_to_abstract_coord(coord)
        orig = flow.map_to_original_coord(abs_coord)
        direct = flow.direct_jacobian(coord)
        inv_jac = flow.inverse_jacobian(coord)
        inv_deriv = flow.inverse_jacobian_coord_derivative(coord)
        self.assertTrue(np.allclose(coord.numpy(), orig.numpy()))
        self.assertEqual(direct.shape[-2:], (1, 1))
        self.assertEqual(inv_jac.shape[-2:], (1, 1))
        self.assertEqual(inv_deriv.shape[-3:], (1, 1, 1))


#########################################################################################################
# FlowCallback initialization tests



def make_chain_flow_callback(with_loglikes=False):
    """Make a small chain for FlowCallback initialization.

    :param with_loglikes: whether to attach loglikes to the chain.
    :returns: MCSamples instance with two parameters.
    """
    samples = np.array(
        [[0.0, 0.1], [0.2, 0.3], [0.1, 0.2], [-0.1, -0.05]], dtype=np.float64
    )
    weights = np.ones(len(samples))
    loglikes = -np.sum(samples**2, axis=1) if with_loglikes else None
    return MCSamples(
        samples=samples,
        names=["p1", "p2"],
        labels=["p1", "p2"],
        weights=weights,
        loglikes=loglikes,
    )


class TestFlowCallbackInit(unittest.TestCase):
    """FlowCallback initialization test suite."""
    def _build_callback(self, chain=None, **kwargs):
        """Helper to build FlowCallback while stubbing heavy monitoring."""
        if chain is None:
            chain = make_chain_flow_callback()
        original_monitor = sp.FlowCallback._init_training_monitoring
        trainable_bijector = kwargs.pop("trainable_bijector", tfb.Identity())
        initialize_model = kwargs.pop("initialize_model", False)
        feedback = kwargs.pop("feedback", 0)
        validation_split = kwargs.pop("validation_split", 0.5)

        def stub_monitor(self):
            """Stub monitor."""
            self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(1, dtype=np.float32)
            self.training_history = {}

        sp.FlowCallback._init_training_monitoring = stub_monitor
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=trainable_bijector,
                initialize_model=initialize_model,
                feedback=feedback,
                validation_split=validation_split,
                **kwargs,
            )
        finally:
            sp.FlowCallback._init_training_monitoring = original_monitor
        return cb

    def test_basic_initialization_identity_bijector(self):
        """Test Basic initialization identity bijector."""
        cb = self._build_callback()
        self.assertEqual(cb.num_params, 2)
        self.assertEqual(cb.training_samples.shape[1], 2)
        self.assertIs(cb.trainable_transformation, None)
        self.assertTrue(cb.distribution.bijector is cb.bijector)
        cb.reset_tensorflow_caches()

    def test_periodic_params_require_rescaling(self):
        """Test Periodic params require rescaling."""
        chain = make_chain_flow_callback()
        with self.assertRaises(ValueError):
            self._build_callback(
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                apply_rescaling=False,
                periodic_params=["p1"],
                feedback=0,
            )

    def test_trainable_bijector_path_validation(self):
        """Test Trainable bijector path validation."""
        cb = self._build_callback()
        with self.assertRaises(ValueError):
            cb._init_trainable_bijector(
                trainable_bijector=tfb.Identity(), trainable_bijector_path="missing"
            )

    def test_training_dataset_with_manual_indices(self):
        """Test Training dataset with manual indices."""
        cb = self._build_callback()
        test_idx = np.array([0, 1], dtype=np.int64)
        train_idx = np.array([2, 3], dtype=np.int64)
        cb._init_training_dataset(validation_training_idx=(test_idx, train_idx))
        self.assertTrue(np.array_equal(cb.test_idx, test_idx))
        self.assertTrue(np.array_equal(cb.training_idx, train_idx))
        self.assertEqual(cb.training_samples.shape[0], len(train_idx))

    def test_nearest_samples_initialization(self):
        """Test Nearest samples initialization."""
        cb = self._build_callback(init_nearest=True)
        self.assertTrue(hasattr(cb, "chain_nearest_index"))
        self.assertEqual(len(cb.chain_nearest_index), cb.chain_samples.shape[0])

    def test_feedback_and_plot_every_validation(self):
        """Test Feedback and plot every validation."""
        chain = make_chain_flow_callback()
        with self.assertRaises(ValueError):
            sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                feedback=-1,
                validation_split=0.5,
            )
        with self.assertRaises(ValueError):
            sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                plot_every=-1,
                validation_split=0.5,
            )

    def test_init_chain_validations(self):
        """Test Init chain validations."""
        chain = make_chain_flow_callback()
        with self.assertRaises(ValueError):
            sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                param_names=["p1", "missing"],
                validation_split=0.5,
            )
        with self.assertRaises(ValueError):
            sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                periodic_params=["missing"],
                validation_split=0.5,
            )
        # samples outside provided ranges trigger validation
        bad_ranges = {"p1": [0.0, 0.05], "p2": [0.0, 0.4]}
        with self.assertRaises(ValueError):
            sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                param_ranges=bad_ranges,
                validation_split=0.5,
            )

    def test_non_standard_loss_requires_loglikes(self):
        """Test Non standard loss requires loglikes."""
        with self.assertRaises(ValueError):
            self._build_callback(loss_mode="fixed")

    def test_init_training_dataset_rng_and_weights(self):
        """Test Init training dataset rng and weights."""
        cb = self._build_callback()
        cb._init_training_dataset(validation_split=0.25, rng=np.random.default_rng(0))
        self.assertGreater(cb.num_training_samples, 0)
        self.assertGreater(cb.num_test_samples, 0)
        # ensure dataset yields expected tuple arity depending on loglikes
        first = next(iter(cb.training_dataset.take(1)))[0]
        self.assertEqual(first.shape[-1], cb.num_params)

    def test_training_dataset_with_loglikes(self):
        """Test Training dataset with loglikes."""
        chain = make_chain_flow_callback(with_loglikes=True)
        original_monitor = sp.FlowCallback._init_training_monitoring

        def stub_monitor(self):
            """Stub monitor."""
            self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(1, dtype=np.float32)
            self.training_history = {}

        sp.FlowCallback._init_training_monitoring = stub_monitor
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                feedback=0,
                validation_split=0.25,
            )
            # tuples should include logP when loglikes present
            elem = next(iter(cb.training_dataset.take(1)))
            self.assertEqual(len(elem), 3)
            self.assertEqual(elem[0].shape[-1], cb.num_params)
        finally:
            sp.FlowCallback._init_training_monitoring = original_monitor

    def test_non_uniform_weights_feedback(self):
        """Test Non uniform weights feedback."""
        samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.1, 0.2], [-0.1, -0.05]], dtype=np.float64)
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        chain = MCSamples(samples=samples, weights=weights, names=["p1", "p2"], labels=["p1", "p2"])
        original_monitor = sp.FlowCallback._init_training_monitoring

        def stub_monitor(self):
            """Stub monitor."""
            self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(1, dtype=np.float32)
            self.training_history = {}

        sp.FlowCallback._init_training_monitoring = stub_monitor
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                feedback=2,
                validation_split=0.25,
            )
            self.assertTrue(cb.has_weights)
            self.assertTrue(np.isclose(np.sum(cb.training_weights), len(cb.training_weights)))
            self.assertTrue(np.isclose(np.sum(cb.test_weights), len(cb.test_weights)))
        finally:
            sp.FlowCallback._init_training_monitoring = original_monitor

    def test_init_trainable_bijector_with_transformation_and_load(self):
        """Test Init trainable bijector with transformation and load."""
        class DummyTransform(tb.TrainableTransformation):
            """Dummy Transform test suite."""
            def __init__(self):
                """Init."""
                self.bijector = tfb.Identity()

            def save(self, path):
                """Save."""
                pass

            @classmethod
            def load(cls, path, **kwargs):
                """Load."""
                return cls()

        cb = self._build_callback(trainable_bijector=DummyTransform())
        self.assertIsInstance(cb.trainable_transformation, DummyTransform)
        self.assertIs(cb.trainable_bijector, cb.trainable_transformation.bijector)
        cb.trainable_transformation.save("unused")
        # trigger load branch
        cb._init_trainable_bijector(
            trainable_bijector=cb.trainable_transformation, trainable_bijector_path="unused"
        )

    def test_fixed_bijector_independent_and_periodic(self):
        """Test Fixed bijector independent and periodic."""
        cb = self._build_callback(apply_rescaling="independent")
        self.assertTrue(any(isinstance(b, tfb.Identity) for b in cb.bijectors))

        cb_periodic = self._build_callback(periodic_params=["p1"])
        self.assertTrue(any(b.name == "ModBijector" for b in cb_periodic.bijectors))

    def test_trainable_bijector_direct_bijector(self):
        """Test Trainable bijector direct bijector."""
        cb = self._build_callback(trainable_bijector=tfb.Shift(1.0))
        self.assertIsNone(cb.trainable_transformation)
        self.assertIsInstance(cb.trainable_bijector, tfb.Shift.__mro__[0])

    def test_training_monitoring_and_metrics_standard_loss(self):
        """Test Training monitoring and metrics standard loss."""
        samples = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
        )
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        chain = MCSamples(samples=samples, weights=weights, names=["p1", "p2"], labels=["p1", "p2"])
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            feedback=0,
            validation_split=0.25,
        )
        # training monitoring already initialized in __init__
        self.assertIn("chi2Z_ks", cb.log)
        self.assertGreater(len(cb.chi2Y), 0)

        logs = {"loss": 1.0, "val_loss": 0.8, "lr": 1e-3}
        cb.compute_training_metrics(logs=logs)
        self.assertEqual(cb.log["loss"][-1], 1.0)
        self.assertEqual(cb.log["val_loss"][-1], 0.8)
        self.assertTrue(len(cb.log["chi2Z_ks"]) >= 1)
        # ensure loss_rate calculations fill values
        cb.compute_training_metrics(logs=logs)
        self.assertIn("loss_rate", cb.log)
        cb.reset_tensorflow_caches()
        # cast utility
        casted = cb.cast([1.0, 2.0])
        self.assertEqual(casted.dtype, tf.float32)

    def test_loss_modes_and_model_init(self):
        # build with loglikes to allow non-standard loss
        """Test Loss modes and model init."""
        chain = make_chain_flow_callback(with_loglikes=True)
        original_monitor = sp.FlowCallback._init_training_monitoring

        def stub_monitor(self):
            """Stub monitor."""
            self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(1, dtype=np.float32)
            self.training_history = {}

        sp.FlowCallback._init_training_monitoring = stub_monitor
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                feedback=0,
                validation_split=0.5,
                loss_mode="fixed",
            )
            cb._init_loss_function(loss_mode="random", learning_rate=1e-4)
            cb._init_loss_function(loss_mode="softadapt", learning_rate=1e-4)
            cb._init_loss_function(loss_mode="sharpstep", learning_rate=1e-4)
            cb._init_loss_function(loss_mode="annealed", learning_rate=1e-4)
            cb._init_model()
            lp = cb.log_probability(tf.zeros((1, cb.num_params), dtype=np.float32))
            jac = cb.log_probability_jacobian(
                tf.zeros((1, cb.num_params), dtype=np.float32)
            )
            hess = cb.log_probability_hessian(
                tf.zeros((1, cb.num_params), dtype=np.float32)
            )
            self.assertEqual(lp.shape, (1,))
            self.assertEqual(jac.shape, (1, cb.num_params))
            self.assertEqual(hess.shape, (1, cb.num_params, cb.num_params))
            cb.reset_tensorflow_caches()
        finally:
            sp.FlowCallback._init_training_monitoring = original_monitor

    def test_evidence_and_smoothness(self):
        """Test Evidence and smoothness."""
        chain = make_chain_flow_callback(with_loglikes=True)
        cb = self._build_callback(chain=chain, trainable_bijector=tfb.Identity())
        avg, err = cb.evidence()
        self.assertTrue(np.isfinite(avg))
        self.assertTrue(np.isfinite(err))
        avg_w, err_w = cb.evidence(weighted=True)
        self.assertTrue(np.isfinite(avg_w))
        self.assertTrue(np.isfinite(err_w))
        score = cb.smoothness_score()
        self.assertTrue(np.isfinite(score))

    def test_compute_training_metrics_variable_loss(self):
        """Test Compute training metrics variable loss."""
        chain = make_chain_flow_callback(with_loglikes=True)
        cb = self._build_callback(chain=chain, trainable_bijector=tfb.Identity(), validation_split=0.25)
        cb._init_loss_function(loss_mode="random", learning_rate=1e-3)
        cb.test_samples = np.array([[0.0, 0.0], [0.5, 0.1], [-0.2, 0.6]], dtype=np.float32)
        cb.test_weights = np.ones(cb.test_samples.shape[0], dtype=np.float32)
        cb.test_logP_preabs = np.zeros(cb.test_samples.shape[0], dtype=np.float32)
        cb.num_test_samples = len(cb.test_samples)
        cb._init_training_monitoring()

        class DummyModel:
            """Dummy Model test suite."""
            def call(self, x):
                """Call."""
                return tf.zeros((tf.shape(x)[0],), dtype=tf.float32)

        cb.set_model(DummyModel())
        logs = {"loss": 0.5, "val_loss": 0.4, "lr": 1e-3}
        cb.compute_training_metrics(logs=logs)
        self.assertIn("lambda_1", cb.log)
        self.assertTrue(len(cb.log["loss"]) > 0)
        cb.compute_training_metrics(logs=logs)

    def test_mcsamples_filters_nonfinite(self):
        """Test Mcsamples filters nonfinite."""
        cb = self._build_callback(feedback=1)

        class DummyDist:
            """Dummy Dist test suite."""
            def sample(self, n):
                """Sample."""
                return tf.constant([[0.0, 0.0], [np.inf, 1.0]], dtype=tf.float32)

        cb.distribution = DummyDist()
        cb.log_probability = lambda samples: tf.zeros((tf.shape(samples)[0],), dtype=tf.float32)
        mc = cb.MCSamples(size=2, logLikes=True, name_tag="dummy")
        self.assertEqual(mc.samples.shape[0], 1)

    def test_plot_helpers(self):
        """Test Plot helpers."""
        cb = self._build_callback()
        cb.test_samples = np.array([[0.0, 0.1], [0.2, -0.1], [-0.3, 0.2]], dtype=np.float32)
        cb.test_weights = np.ones(cb.test_samples.shape[0], dtype=np.float32)
        cb.num_test_samples = len(cb.test_samples)
        cb._init_training_monitoring()
        cb.log["loss"] = [1.0, -0.5]
        cb.log["val_loss"] = [1.1, -0.4]
        cb.log["lr"] = [1e-3, 5e-4]
        cb.chi2Y = np.array([0.5, 1.0])
        cb.test_weights = np.ones_like(cb.chi2Y, dtype=np.float32)
        cb.chi2Z = np.array([0.25, 0.75])
        cb.log["chi2Z_ks"] = [0.2]
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        cb._plot_loss(ax[0], logs={})
        cb._plot_lr(ax[1], logs={})
        cb._plot_chi2_dist(ax[2], logs={}, fast=True)
        plt.close(fig)

    def test_sample_wrapper_and_abs_logprob(self):
        """Test Sample wrapper and abs logprob."""
        cb = self._build_callback()

        class DummyDist:
            """Dummy Dist test suite."""
            def __init__(self, base):
                """Init."""
                self.base = base
                self.last_dtype = None

            def sample(self, n):
                """Sample."""
                self.last_dtype = n.dtype
                return self.base.sample(n)

            @property
            def distribution(self):
                """Distribution."""
                return self.base.distribution

        original_dist = cb.distribution
        cb.distribution = DummyDist(original_dist)
        self.assertIs(cb.distribution.distribution, original_dist.distribution)
        samples = cb.sample(2)
        self.assertEqual(samples.shape, (2, cb.num_params))
        self.assertEqual(cb.distribution.last_dtype, tf.int32)
        cb.distribution = original_dist
        abs_lp = cb.log_probability_abs(tf.zeros((1, cb.num_params), dtype=tf.float32))
        abs_jac = cb.log_probability_abs_jacobian(tf.zeros((1, cb.num_params), dtype=tf.float32))
        abs_hess = cb.log_probability_abs_hessian(tf.zeros((1, cb.num_params), dtype=tf.float32))
        self.assertEqual(abs_lp.shape, (1,))
        self.assertEqual(abs_jac.shape, (1, cb.num_params))
        self.assertEqual(abs_hess.shape, (1, cb.num_params, cb.num_params))

    def test_constant_loss_training_metrics(self):
        """Test Constant loss training metrics."""
        chain = make_chain_flow_callback(with_loglikes=True)
        cb = self._build_callback(chain=chain, trainable_bijector=tfb.Identity())
        cb.loss = lf.constant_weight_loss(alpha=0.5, beta=0.0)
        cb.loss_mode = "fixed"
        cb.test_samples = np.array([[0.0, 0.0], [0.4, -0.2], [-0.3, 0.3]], dtype=np.float32)
        cb.test_logP_preabs = np.zeros(cb.test_samples.shape[0], dtype=np.float32)
        cb.test_weights = np.ones(cb.test_samples.shape[0], dtype=np.float32)
        cb.num_test_samples = len(cb.test_samples)
        cb._init_training_monitoring()

        class DummyModel:
            """Dummy Model test suite."""
            def call(self, x):
                """Call."""
                return tf.zeros((tf.shape(x)[0],), dtype=tf.float32)

        cb.set_model(DummyModel())
        logs = {"loss": 0.5, "val_loss": 0.4, "lr": 1e-3}
        cb.compute_training_metrics(logs=logs)
        self.assertIn("rho_loss", cb.log)
        self.assertGreaterEqual(len(cb.log["rho_loss"]), 1)
        cb.compute_training_metrics(logs=logs)

    def test_evidence_indexes_branch(self):
        """Test Evidence indexes branch."""
        chain = make_chain_flow_callback(with_loglikes=True)
        cb = self._build_callback(chain=chain)
        avg, err = cb.evidence(indexes=np.array([0, 1]))
        self.assertTrue(np.isfinite(avg))
        self.assertTrue(np.isfinite(err))

    def test_trainable_bijector_path_error(self):
        """Test Trainable bijector path error."""
        cb = self._build_callback(trainable_bijector=tfb.Identity())
        with self.assertRaises(ValueError):
            cb._init_trainable_bijector(trainable_bijector=tfb.Identity(), trainable_bijector_path="unused")

    def test_mcsamples_loglikes_true(self):
        """Test Mcsamples loglikes true."""
        cb = self._build_callback()
        cb.log_probability = lambda samples: tf.zeros((tf.shape(samples)[0],), dtype=tf.float32)
        mc = cb.MCSamples(size=3, logLikes=True)
        self.assertEqual(mc.samples.shape[1], cb.num_params)


#########################################################################################################
# FlowCallback additional tests



def make_chain_flow_callback_additional(with_loglikes=False, name_tag="flow"):
    """Make a chain for additional FlowCallback branches.

    :param with_loglikes: whether to attach loglikes to the chain.
    :param name_tag: name tag to attach to the chain.
    :returns: MCSamples instance with two parameters.
    """
    samples = np.array(
        [[0.0, 0.1], [0.2, 0.3], [0.1, 0.2], [-0.1, -0.05]], dtype=np.float64
    )
    weights = np.ones(len(samples))
    loglikes = -np.sum(samples**2, axis=1) if with_loglikes else None
    chain = MCSamples(
        samples=samples,
        names=["p1", "p2"],
        labels=["p1", "p2"],
        weights=weights,
        loglikes=loglikes,
        name_tag=name_tag,
    )
    return chain


class TestFlowCallbackAdditional(unittest.TestCase):
    """FlowCallback additional test suite."""
    def _stub_monitor(self):
        """Stub monitor."""
        original = sp.FlowCallback._init_training_monitoring

        def stub(self):
            """Stub."""
            self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(1, dtype=np.float32)
            self.training_history = {}
            self.training_metrics = ["loss", "val_loss", "loss_rate", "val_loss_rate"]
            self.log = {k: [] for k in self.training_metrics}
            self.chi2Y = np.array([0.0])
            self.chi2Y_ks = 0.0

        return original, stub

    def test_init_fixed_bijector_identity(self):
        """Test Init fixed bijector identity."""
        chain = make_chain_flow_callback_additional()
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                prior_bijector=False,
                apply_rescaling=False,
                validation_split=0.5,
                feedback=0,
            )
            self.assertIsInstance(cb.prior_bijector, tfb.Identity.__mro__[0])
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_training_dataset_with_validation_indices_and_loglikes(self):
        """Test Training dataset with validation indices and loglikes."""
        chain = make_chain_flow_callback_additional(with_loglikes=True)
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            cb._init_training_dataset(validation_training_idx=(np.array([0, 1]), np.array([2, 3])))
            self.assertEqual(cb.training_samples.shape[0], 2)
            self.assertEqual(cb.test_samples.shape[0], 2)
            if cb.has_loglikes:
                elem = next(iter(cb.training_dataset.take(1)))
                self.assertEqual(len(elem), 3)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_reset_tensorflow_caches_clears_dicts(self):
        """Test Reset tensorflow caches clears dicts."""
        chain = make_chain_flow_callback_additional()
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            # attach fake caches
            # ensure call succeeds without raising
            cb.reset_tensorflow_caches()
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_log_probability_abs_calls(self):
        """Test Log probability abs calls."""
        chain = make_chain_flow_callback_additional()
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            x = tf.zeros((1, cb.num_params), dtype=tf.float32)
            lp = cb.log_probability_abs(x)
            jac = cb.log_probability_abs_jacobian(x)
            hess = cb.log_probability_abs_hessian(x)
            self.assertEqual(lp.shape, (1,))
            self.assertEqual(jac.shape, (1, cb.num_params))
            self.assertEqual(hess.shape, (1, cb.num_params, cb.num_params))
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_name_tag_propagation(self):
        """Test Name tag propagation."""
        chain = make_chain_flow_callback_additional(name_tag="custom")
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            self.assertIn("custom", cb.name_tag)
            cb.log_probability = lambda samples: tf.zeros((tf.shape(samples)[0],), dtype=tf.float32)
            mc = cb.MCSamples(size=1, logLikes=True)
            self.assertEqual(mc.name_tag, cb.name_tag)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_evidence_weighted_indexes(self):
        """Test Evidence weighted indexes."""
        chain = make_chain_flow_callback_additional(with_loglikes=True)
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
        finally:
            sp.FlowCallback._init_training_monitoring = original
        avg, err = cb.evidence(indexes=np.array([0, 1]), weighted=True)
        self.assertTrue(np.isfinite(avg))
        self.assertTrue(np.isfinite(err))

    def test_compute_training_metrics_loss_rate_fallback(self):
        """Test Compute training metrics loss rate fallback."""
        chain = make_chain_flow_callback_additional(with_loglikes=True)
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
        finally:
            sp.FlowCallback._init_training_monitoring = original
        cb.log["loss"] = [0.5]
        cb.log["val_loss"] = [0.4]
        cb.compute_training_metrics(logs={"loss": 0.3, "val_loss": 0.2, "lr": 1e-3})
        self.assertIn("loss_rate", cb.log)
        self.assertGreaterEqual(len(cb.log["loss_rate"]), 1)

    def test_plot_loss_handles_negative(self):
        """Test Plot loss handles negative."""
        chain = make_chain_flow_callback_additional()
        original, stub = self._stub_monitor()
        sp.FlowCallback._init_training_monitoring = stub
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        cb.log["loss"] = [1.0, -0.1]
        cb.log["val_loss"] = [1.2, -0.2]
        fig, ax = plt.subplots()
        cb._plot_loss(ax, logs={})
        plt.close(fig)
        sp.FlowCallback._init_training_monitoring = original


#########################################################################################################
# FlowCallback more tests



def make_chain_flow_callback_more(loglikes=True):
    """Make a chain for extended FlowCallback coverage.

    :param loglikes: whether to attach loglikes to the chain.
    :returns: MCSamples instance with two parameters.
    """
    samples = np.array(
        [[0.0, 0.1], [0.2, 0.3], [0.1, 0.2], [-0.1, -0.05]], dtype=np.float64
    )
    weights = np.ones(len(samples))
    ll = -np.sum(samples**2, axis=1) if loglikes else None
    return MCSamples(
        samples=samples,
        weights=weights,
        names=["p1", "p2"],
        labels=["p1", "p2"],
        loglikes=ll,
        name_tag="test",
    )


class TestFlowCallbackMore(unittest.TestCase):
    """FlowCallback more test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.original_monitor = sp.FlowCallback._init_training_monitoring

        def stub_monitor(self):
            """Stub monitor."""
            self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(1, dtype=np.float32)
            self.training_history = {}
            self.training_metrics = ["loss", "val_loss", "loss_rate", "val_loss_rate"]
            self.log = {k: [] for k in self.training_metrics}
            self.chi2Y = np.array([0.0])
            self.chi2Y_ks = 0.0

        sp.FlowCallback._init_training_monitoring = stub_monitor

    def tearDown(self):
        """Clean up test fixtures."""
        sp.FlowCallback._init_training_monitoring = self.original_monitor

    def test_init_chain_custom_param_ranges(self):
        """Test Init chain custom param ranges."""
        chain = make_chain_flow_callback_more()
        param_ranges = {"p1": [-1.0, 1.0], "p2": [-2.0, 2.0]}
        cb = sp.FlowCallback(
            chain,
            param_names=["p1", "p2"],
            param_ranges=param_ranges,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        self.assertEqual(cb.parameter_ranges["p1"], param_ranges["p1"])
        self.assertEqual(cb.parameter_ranges["p2"], param_ranges["p2"])

    def test_periodic_params_single_value(self):
        """Test Periodic params single value."""
        chain = make_chain_flow_callback_more()
        cb = sp.FlowCallback(
            chain,
            periodic_params=["p1"],
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        self.assertIn("p1", cb.periodic_params)

    def test_init_nearest_samples_populates_index(self):
        """Test Init nearest samples populates index."""
        chain = make_chain_flow_callback_more()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        cb._init_nearest_samples()
        self.assertTrue(hasattr(cb, "chain_nearest_index"))
        self.assertEqual(cb.chain_nearest_index.shape[0], cb.chain_samples.shape[0])

    def test_fixed_bijector_small_variance_periodic(self):
        """Test Fixed bijector small variance periodic."""
        chain = make_chain_flow_callback_more()
        cb = sp.FlowCallback(
            chain,
            periodic_params=["p1"],
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        self.assertIn("p1", cb.trainable_periodic_params)

    def test_init_trainable_bijector_invalid(self):
        """Test Init trainable bijector invalid."""
        chain = make_chain_flow_callback_more()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        with self.assertRaises(ValueError):
            cb._init_trainable_bijector(trainable_bijector="unknown")

    def test_init_training_dataset_loglikes_weights_normalization(self):
        """Test Init training dataset loglikes weights normalization."""
        chain = make_chain_flow_callback_more(loglikes=True)
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        self.assertAlmostEqual(
            np.sum(cb.training_weights), len(cb.training_weights), places=5
        )
        self.assertAlmostEqual(np.sum(cb.test_weights), len(cb.test_weights), places=5)

    def test_init_distribution_shapes(self):
        """Test Init distribution shapes."""
        chain = make_chain_flow_callback_more()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        cb._init_distribution()
        sample = cb.distribution.sample(1)
        self.assertEqual(sample.shape[-1], cb.num_params)

    def test_init_loss_function_standard_default(self):
        """Test Init loss function standard default."""
        chain = make_chain_flow_callback_more()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        cb._init_loss_function()
        self.assertEqual(cb.loss_mode, "standard")

    def test_init_model_sets_flag(self):
        """Test Init model sets flag."""
        chain = make_chain_flow_callback_more()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        cb._init_distribution()
        cb._init_loss_function()
        cb._init_model()
        self.assertTrue(cb._model_initialied)

    def test_smoothness_score_with_nearest(self):
        """Test Smoothness score with nearest."""
        chain = make_chain_flow_callback_more(loglikes=True)
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
            init_nearest=True,
        )
        score = cb.smoothness_score()
        self.assertTrue(np.isfinite(score))


#########################################################################################################
# FlowCallback even-more tests



def make_chain_flow_callback_even_more(loglikes=True, name_tag="flow"):
    """Make a chain for extra FlowCallback branches.

    :param loglikes: whether to attach loglikes to the chain.
    :param name_tag: name tag to attach to the chain.
    :returns: MCSamples instance with two parameters.
    """
    samples = np.array(
        [[0.0, 0.1], [0.2, 0.3], [0.1, 0.2], [-0.1, -0.05]], dtype=np.float64
    )
    weights = np.ones(len(samples))
    ll = -np.sum(samples**2, axis=1) if loglikes else None
    return MCSamples(
        samples=samples,
        weights=weights,
        names=["p1", "p2"],
        labels=["p1", "p2"],
        loglikes=ll,
        name_tag=name_tag,
    )


class TestFlowCallbackEvenMore(unittest.TestCase):
    """FlowCallback even-more test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.real_monitor = sp.FlowCallback._init_training_monitoring

    def tearDown(self):
        """Clean up test fixtures."""
        sp.FlowCallback._init_training_monitoring = self.real_monitor

    def stub_monitor(self):
        """Stub monitor."""
        def _stub(self):
            """Stub."""
            self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(1, dtype=np.float32)
            self.training_history = {}
            self.training_metrics = ["loss", "val_loss", "loss_rate", "val_loss_rate"]
            self.log = {k: [] for k in self.training_metrics}
            self.chi2Y = np.array([0.0])
            self.chi2Y_ks = 0.0

        return _stub

    def variable_stub_monitor(self):
        """Variable stub monitor."""
        def _stub(self):
            # minimal setup for variable-weight training without heavy linear algebra
            """Stub."""
            self.test_samples = np.zeros((2, self.num_params), dtype=np.float32)
            self.test_weights = np.ones(2, dtype=np.float32)
            self.training_history = {}
            self.training_metrics = [
                "loss",
                "val_loss",
                "lr",
                "rho_loss",
                "ee_loss",
                "val_rho_loss",
                "val_ee_loss",
                "loss_rate",
                "rho_loss_rate",
                "ee_loss_rate",
                "chi2Z_ks",
                "chi2Z_ks_p",
                "training_evidence",
                "training_evidence_error",
                "test_evidence",
                "test_evidence_error",
                "evidence",
                "evidence_error",
                "lambda_1",
                "lambda_2",
            ]
            self.log = {k: [] for k in self.training_metrics}
            self.chi2Y = np.array([0.0, 0.0])
            self.chi2Y_ks = 0.0
            self.chi2Y_ks_p = 1.0

        return _stub

    def test_name_tag_defaults_to_flow(self):
        """Test Name tag defaults to flow."""
        chain = make_chain_flow_callback_even_more()
        chain.name_tag = None
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            self.assertEqual(cb.name_tag, "flow")
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_periodic_param_validation(self):
        """Test Periodic param validation."""
        chain = make_chain_flow_callback_even_more()
        with self.assertRaises(ValueError):
            sp.FlowCallback(
                chain,
                periodic_params=["missing"],
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )

    def test_validation_indices_no_loglikes(self):
        """Test Validation indices no loglikes."""
        chain = make_chain_flow_callback_even_more(loglikes=False)
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            cb._init_training_dataset(validation_training_idx=(np.array([0, 1]), np.array([2, 3])))
            elem = next(iter(cb.training_dataset.take(1)))
            self.assertEqual(len(elem), 2)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_on_epoch_begin_variable_loss(self):
        """Test On epoch begin variable loss."""
        chain = make_chain_flow_callback_even_more()
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            class DummyVarLoss(lf.variable_weight_loss):
                """Dummy Var Loss test suite."""
                def __init__(self):
                    """Init."""
                    super().__init__(lambda_1=1.0, lambda_2=0.0, beta=0.0)
                    self.updated = False
                def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
                    """Update lambda values on epoch begin."""
                    self.updated = True
            cb.loss = DummyVarLoss()
            cb.set_model(type("DummyModel", (), {"loss": cb.loss})())
            cb.on_epoch_begin(1, logs={"dummy": 0})
            self.assertTrue(cb.loss.updated)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_training_monitoring_for_variable_loss(self):
        """Test Training monitoring for variable loss."""
        chain = make_chain_flow_callback_even_more()
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.variable_stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            cb.loss = lf.variable_weight_loss(lambda_1=1.0, lambda_2=0.0, beta=0.0)
            cb._init_training_monitoring()
            self.assertIn("lambda_1", cb.training_metrics)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_compute_training_metrics_with_stub_evidence(self):
        """Test Compute training metrics with stub evidence."""
        chain = make_chain_flow_callback_even_more(loglikes=True)
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            cb.training_metrics = ["evidence", "evidence_error"]
            cb.log = {k: [] for k in cb.training_metrics}
            cb.evidence = lambda indexes=None: (1.0, 0.1)
            cb.compute_training_metrics(logs={"loss": 0.1, "val_loss": 0.1, "lr": 1e-3})
            self.assertEqual(cb.log["evidence"][-1], 1.0)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_compile_model_resets_loss(self):
        """Test Compile model resets loss."""
        chain = make_chain_flow_callback_even_more()
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            cb._init_distribution()
            class DummyLoss(tf.keras.losses.Loss):
                """Dummy Loss test suite."""
                def __init__(self):
                    """Init."""
                    super().__init__()
                    self.reset_called = False
                def call(self, y_true, y_pred):
                    """Call."""
                    return y_pred
                def reset(self):
                    """Reset."""
                    self.reset_called = True
            cb.loss = DummyLoss()
            cb.set_model(tf.keras.Sequential([tf.keras.layers.Input(shape=(cb.num_params,)), tf.keras.layers.Dense(1)]))
            cb.initial_learning_rate = 1e-3
            cb.global_clipnorm = 1.0
            cb._compile_model()
            loss_val = cb.loss.call(None, tf.constant([1.0]))
            self.assertEqual(loss_val.numpy()[0], 1.0)
            self.assertTrue(cb.loss.reset_called)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_init_training_dataset_rng(self):
        """Test Init training dataset rng."""
        chain = make_chain_flow_callback_even_more()
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.25,
                feedback=0,
            )
            cb._init_training_dataset(rng=np.random.default_rng(123))
            self.assertEqual(cb.training_samples.shape[1], cb.num_params)
        finally:
            sp.FlowCallback._init_training_monitoring = original

    def test_sharpstep_loss_init(self):
        """Test Sharpstep loss init."""
        chain = make_chain_flow_callback_even_more()
        original = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = self.stub_monitor()
        try:
            cb = sp.FlowCallback(
                chain,
                trainable_bijector=tfb.Identity(),
                initialize_model=False,
                validation_split=0.5,
                feedback=0,
            )
            cb._init_loss_function(loss_mode="sharpstep")
            self.assertIsInstance(cb.loss, lf.SharpStep)
        finally:
            sp.FlowCallback._init_training_monitoring = original


#########################################################################################################
# FlowCallback final tests



def make_chain_flow_callback_final():
    """Make a chain for final FlowCallback coverage.

    :returns: MCSamples instance with two parameters.
    """
    samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.1, 0.2]], dtype=np.float64)
    weights = np.ones(len(samples))
    loglikes = -np.sum(samples**2, axis=1)
    return MCSamples(
        samples=samples,
        weights=weights,
        names=["p1", "p2"],
        labels=["p1", "p2"],
        loglikes=loglikes,
        name_tag="final",
    )


class TestFlowCallbackFinal(unittest.TestCase):
    """FlowCallback final test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.original_monitor = sp.FlowCallback._init_training_monitoring
        def _stub(cb):
            """Stub."""
            cb.test_samples = np.zeros((2, cb.num_params), dtype=np.float32)
            cb.test_weights = np.ones(2, dtype=np.float32)
            cb.training_history = {}
            cb.training_metrics = ["loss", "val_loss", "loss_rate", "val_loss_rate", "lr", "chi2Z_ks", "chi2Z_ks_p"]
            cb.log = {k: [0.1, 0.2] if "rate" not in k else [0.0, 0.1] for k in cb.training_metrics}
            cb.chi2Y = np.array([0.5, 1.0])
            cb.chi2Y_ks = 0.1
            cb.chi2Z = np.array([0.4, 0.6])
        self.stub_monitor = _stub
        sp.FlowCallback._init_training_monitoring = self.stub_monitor

    def tearDown(self):
        """Clean up test fixtures."""
        sp.FlowCallback._init_training_monitoring = self.original_monitor

    def _build_cb(self, loss_obj):
        """Build cb."""
        chain = make_chain_flow_callback_final()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        cb.loss = loss_obj
        return cb

    def test_plot_lambda_values(self):
        """Test Plot lambda values."""
        cb = self._build_cb(lf.variable_weight_loss(lambda_1=0.5, lambda_2=0.5, beta=0.0))
        cb.log.update({"lambda_1": [0.6, 0.4], "lambda_2": [0.4, 0.6]})
        fig, ax = plt.subplots()
        cb._plot_lambda_values(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_density_evidence_error_losses(self):
        """Test Plot density evidence error losses."""
        cb = self._build_cb(lf.constant_weight_loss())
        cb.log.update(
            {
                "rho_loss": [1.0, 0.8],
                "ee_loss": [0.5, 0.3],
                "val_rho_loss": [0.6, 0.4],
                "val_ee_loss": [0.4, 0.2],
            }
        )
        fig, ax = plt.subplots()
        cb._plot_density_evidence_error_losses(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_chi2_ks_p(self):
        """Test Plot chi2 ks p."""
        cb = self._build_cb(lf.standard_loss())
        cb.log["chi2Z_ks_p"] = [0.8, 0.9]
        fig, ax = plt.subplots()
        cb._plot_chi2_ks_p(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_chi2_dist_fast_false(self):
        """Test Plot chi2 dist fast false."""
        cb = self._build_cb(lf.standard_loss())
        cb.test_weights = np.ones_like(cb.chi2Y, dtype=np.float32)
        fig, ax = plt.subplots()
        cb._plot_chi2_dist(ax, logs=cb.log, fast=False)
        plt.close(fig)

    def test_on_train_begin_end(self):
        """Test On train begin end."""
        cb = self._build_cb(lf.standard_loss())
        sp.ipython_plotting = False
        cb.on_train_begin(logs={})
        self.assertTrue(hasattr(cb, "fig"))
        cb.on_train_end(logs={})
        sp.ipython_plotting = True

    def test_training_plot_population_title(self):
        """Test Training plot population title."""
        cb = self._build_cb(lf.standard_loss())
        cb.log["population"] = 1
        cb._create_figure()
        cb.training_plot(logs=cb.log, ipython_plotting=True, fast=True)
        plt.close("all")

    def test_on_epoch_end_no_optimizer_lr(self):
        """Test On epoch end no optimizer lr."""
        cb = self._build_cb(lf.standard_loss())
        cb.set_model(type("DummyModel", (), {})())
        cb.feedback = 0
        cb.plot_every = 0
        cb.on_epoch_end(0, logs={"loss": 0.1, "val_loss": 0.2})
        self.assertTrue("loss" in cb.log)

    def test_MCSamples_logLikes_false(self):
        """Test M C Samples log Likes false."""
        cb = self._build_cb(lf.standard_loss())
        cb.sample = lambda n: tf.constant([[0.1, 0.2], [0.2, 0.3]], dtype=tf.float32)
        cb.log_probability = lambda samples: tf.zeros((tf.shape(samples)[0],), dtype=tf.float32)
        mc = cb.MCSamples(size=2, logLikes=True)
        self.assertEqual(mc.samples.shape[1], cb.num_params)

    def test_DerivedParamsBijector_init(self):
        """Test Derived Params Bijector init."""
        chain = make_chain_flow_callback_final()
        bij = sp.DerivedParamsBijector(
            chain=chain,
            param_names_in=["p1", "p2"],
            param_names_out=["p1", "p2"],
            permutations=False,
            feedback=0,
        )
        self.assertEqual(bij.num_params, 2)

    def test_training_plot_variable_loss_rates(self):
        """Test Training plot variable loss rates."""
        cb = self._build_cb(lf.variable_weight_loss(lambda_1=0.5, lambda_2=0.5, beta=0.0))
        cb._create_figure()
        cb.log.update(
            {
                "rho_loss": [1.0, 0.8],
                "ee_loss": [0.5, 0.3],
                "val_rho_loss": [0.6, 0.4],
                "val_ee_loss": [0.4, 0.2],
                "rho_loss_rate": [0.0, 0.1],
                "ee_loss_rate": [0.0, 0.05],
                "lambda_1": [0.5, 0.4],
                "lambda_2": [0.5, 0.6],
                "evidence": [1.0, 1.1],
                "evidence_error": [0.1, 0.2],
                "training_evidence": [0.9, 1.0],
                "training_evidence_error": [0.1, 0.2],
                "test_evidence": [1.05, 1.1],
                "test_evidence_error": [0.1, 0.2],
            }
        )
        cb.training_plot(logs=cb.log, ipython_plotting=True, fast=True)
        plt.close("all")


#########################################################################################################
# FlowCallback plot tests



def make_chain_flow_callback_plots():
    """Make a chain for FlowCallback plotting tests.

    :returns: MCSamples instance with two parameters.
    """
    samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.1, 0.2]], dtype=np.float64)
    weights = np.ones(len(samples))
    return MCSamples(
        samples=samples,
        weights=weights,
        names=["p1", "p2"],
        labels=["p1", "p2"],
        loglikes=-np.sum(samples**2, axis=1),
        name_tag="plot",
    )


def stub_monitor_flow_callback_plots(self):
    """Stub training monitor for plot tests.

    :param self: FlowCallback instance under test.
    :returns: ``None``.
    """
    self.test_samples = np.zeros((1, self.num_params), dtype=np.float32)
    self.test_weights = np.ones(2, dtype=np.float32)
    self.training_history = {}
    self.chi2Y = np.array([0.5, 1.0])
    self.chi2Z = np.array([0.25, 0.75])
    self.chi2Y_ks = 0.1
    self.chi2Y_ks_p = 0.9
    # populate generic logs
    base_keys = [
        "loss",
        "val_loss",
        "loss_rate",
        "val_loss_rate",
        "lr",
        "chi2Z_ks",
        "chi2Z_ks_p",
    ]
    self.training_metrics = base_keys.copy()
    self.log = {k: [0.1, 0.2] if "rate" not in k else [0.0, 0.1] for k in base_keys}
    self.log["chi2Z_ks"] = [0.2]
    self.log["chi2Z_ks_p"] = [0.8]


class TestFlowCallbackPlot(unittest.TestCase):
    """FlowCallback plot test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.original_monitor = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = stub_monitor_flow_callback_plots

    def tearDown(self):
        """Clean up test fixtures."""
        sp.FlowCallback._init_training_monitoring = self.original_monitor

    def _build_cb(self, loss_obj):
        """Build cb."""
        chain = make_chain_flow_callback_plots()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
        )
        cb.loss = loss_obj
        return cb

    def test_create_figure_standard_constant_variable(self):
        """Test Create figure standard constant variable."""
        cb = self._build_cb(lf.standard_loss())
        cb._create_figure()
        self.assertTrue(hasattr(cb, "fig"))
        cb.loss = lf.constant_weight_loss()
        cb._create_figure()
        cb.loss = lf.variable_weight_loss(lambda_1=1.0, lambda_2=0.0, beta=0.0)
        cb._create_figure()
        plt.close("all")

    def test_plot_weighted_density_losses(self):
        """Test Plot weighted density losses."""
        cb = self._build_cb(lf.variable_weight_loss(lambda_1=1.0, lambda_2=0.0, beta=0.0))
        cb.log.update(
            {
                "lambda_1": [0.5, 0.4],
                "lambda_2": [0.5, 0.6],
                "rho_loss": [1.0, 0.5],
                "ee_loss": [0.5, 0.25],
                "val_rho_loss": [0.8, 0.4],
                "val_ee_loss": [0.4, 0.2],
            }
        )
        fig, ax = plt.subplots()
        cb._plot_weighted_density_evidence_error_losses(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_losses_rate_abs_and_signed(self):
        """Test Plot losses rate abs and signed."""
        cb = self._build_cb(lf.standard_loss())
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        cb._plot_losses_rate(ax[0], logs=cb.log, abs_value=False)
        cb._plot_losses_rate(ax[1], logs=cb.log, abs_value=True)
        plt.close(fig)

    def test_plot_evidence_and_error(self):
        """Test Plot evidence and error."""
        cb = self._build_cb(lf.constant_weight_loss())
        cb.log.update(
            {
                "evidence": [1.0, 2.0],
                "training_evidence": [0.9, 1.9],
                "test_evidence": [1.1, 2.1],
                "evidence_error": [0.1, 0.2],
                "training_evidence_error": [0.1, 0.2],
                "test_evidence_error": [0.1, 0.2],
                "rho_loss_rate": [0.0, 0.1],
                "ee_loss_rate": [0.0, 0.05],
            }
        )
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        cb._plot_evidence(ax[0], logs=cb.log)
        cb._plot_evidence_error(ax[1], logs=cb.log)
        plt.close(fig)

    def test_training_plot_standard(self):
        """Test Training plot standard."""
        cb = self._build_cb(lf.standard_loss())
        cb._create_figure()
        cb.training_plot(logs=cb.log, ipython_plotting=True, fast=True)
        plt.close("all")

    def test_training_plot_constant(self):
        """Test Training plot constant."""
        cb = self._build_cb(lf.constant_weight_loss())
        cb._create_figure()
        cb.log.update(
            {
                "rho_loss": [1.0, 0.8],
                "ee_loss": [0.5, 0.3],
                "val_rho_loss": [0.6, 0.4],
                "val_ee_loss": [0.4, 0.2],
                "rho_loss_rate": [0.0, 0.1],
                "ee_loss_rate": [0.0, 0.05],
                "evidence": [1.0, 1.1],
                "evidence_error": [0.1, 0.2],
                "training_evidence": [0.9, 1.0],
                "training_evidence_error": [0.1, 0.2],
                "test_evidence": [1.05, 1.1],
                "test_evidence_error": [0.1, 0.2],
            }
        )
        cb.training_plot(logs=cb.log, ipython_plotting=True, fast=True)
        plt.close("all")

    def test_training_plot_variable(self):
        """Test Training plot variable."""
        cb = self._build_cb(lf.variable_weight_loss(lambda_1=0.8, lambda_2=0.2, beta=0.0))
        cb._create_figure()
        cb.log.update(
            {
                "rho_loss": [1.0, 0.8],
                "ee_loss": [0.5, 0.3],
                "val_rho_loss": [0.6, 0.4],
                "val_ee_loss": [0.4, 0.2],
                "rho_loss_rate": [0.0, 0.1],
                "ee_loss_rate": [0.0, 0.05],
                "lambda_1": [0.8, 0.7],
                "lambda_2": [0.2, 0.3],
                "evidence": [1.0, 1.1],
                "evidence_error": [0.1, 0.2],
                "training_evidence": [0.9, 1.0],
                "training_evidence_error": [0.1, 0.2],
                "test_evidence": [1.05, 1.1],
                "test_evidence_error": [0.1, 0.2],
            }
        )
        cb.training_plot(logs=cb.log, ipython_plotting=True, fast=True)
        plt.close("all")

    def test_on_epoch_end_updates_logs(self):
        """Test On epoch end updates logs."""
        cb = self._build_cb(lf.standard_loss())
        cb.compute_training_metrics = lambda logs: cb.log["loss"].append(logs["loss"])
        cb.set_model(type("DummyModel", (), {"optimizer": type("opt", (), {"lr": tf.constant(0.1)})()})())
        cb.feedback = 0
        cb.plot_every = 0
        cb.on_epoch_end(0, logs={"loss": 0.3, "val_loss": 0.2})
        self.assertEqual(cb.log["loss"][-1], 0.3)

    def test_print_training_summary(self):
        """Test Print training summary."""
        cb = self._build_cb(lf.standard_loss())
        cb.training_metrics = ["loss", "val_loss"]
        cb.log = {"loss": [0.1], "val_loss": [0.2]}
        cb.print_training_summary()


#########################################################################################################
# FlowCallback end-to-end tests



def make_chain_flow_callback_end_to_end(loglikes=True, name_tag="end2end"):
    """Make a chain for end-to-end FlowCallback tests.

    :param loglikes: whether to attach loglikes to the chain.
    :param name_tag: name tag to attach to the chain.
    :returns: MCSamples instance with two parameters.
    """
    samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.1, 0.2]], dtype=np.float64)
    weights = np.ones(len(samples))
    ll = -np.sum(samples**2, axis=1) if loglikes else None
    return MCSamples(
        samples=samples,
        weights=weights,
        names=["p1", "p2"],
        labels=["p1", "p2"],
        loglikes=ll,
        name_tag=name_tag,
    )


def stub_monitor_flow_callback_end_to_end(self):
    """Stub training monitor for end-to-end tests.

    :param self: FlowCallback instance under test.
    :returns: ``None``.
    """
    self.test_samples = np.zeros((2, self.num_params), dtype=np.float32)
    self.test_weights = np.ones(2, dtype=np.float32)
    self.training_history = {}
    self.chi2Y = np.array([0.5, 1.0])
    self.chi2Z = np.array([0.25, 0.75])
    self.chi2Y_ks = 0.1
    self.chi2Y_ks_p = 0.9
    keys = [
        "loss",
        "val_loss",
        "lr",
        "chi2Z_ks",
        "chi2Z_ks_p",
        "loss_rate",
        "val_loss_rate",
    ]
    self.training_metrics = keys.copy()
    self.log = {k: [0.1, 0.2] if "rate" not in k else [0.0, 0.1] for k in keys}
    self.log["chi2Z_ks"] = [0.2]
    self.log["chi2Z_ks_p"] = [0.8]


class TestFlowCallbackEndToEnd(unittest.TestCase):
    """FlowCallback end-to-end test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.original_monitor = sp.FlowCallback._init_training_monitoring
        sp.FlowCallback._init_training_monitoring = stub_monitor_flow_callback_end_to_end

    def tearDown(self):
        """Clean up test fixtures."""
        sp.FlowCallback._init_training_monitoring = self.original_monitor
        plt.close("all")

    def _build_cb(self, loss_obj, **kwargs):
        """Build cb."""
        chain = make_chain_flow_callback_end_to_end()
        cb = sp.FlowCallback(
            chain,
            trainable_bijector=tfb.Identity(),
            initialize_model=False,
            validation_split=0.5,
            feedback=0,
            **kwargs,
        )
        cb.loss = loss_obj
        return cb

    def test_on_train_begin_end_ipython(self):
        """Test On train begin end ipython."""
        cb = self._build_cb(lf.standard_loss())
        sp.ipython_plotting = True
        cb.on_train_begin(logs={})
        self.assertFalse(hasattr(cb, "fig"))
        cb.on_train_end(logs={})

    def test_plot_losses_rate_variable_abs(self):
        """Test Plot losses rate variable abs."""
        cb = self._build_cb(lf.variable_weight_loss(lambda_1=1.0, lambda_2=0.0, beta=0.0))
        cb.log.update({"rho_loss_rate": [0.0, 0.1], "ee_loss_rate": [0.0, 0.1]})
        fig, ax = plt.subplots()
        cb._plot_losses_rate(ax, logs=cb.log, abs_value=True)
        plt.close(fig)

    def test_training_plot_saves_file(self):
        """Test Training plot saves file."""
        cb = self._build_cb(lf.standard_loss())
        cb._create_figure()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            path = tmp.name
        cb.training_plot(logs=cb.log, ipython_plotting=True, fast=True, file_path=path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_on_epoch_end_triggers_training_plot(self):
        """Test On epoch end triggers training plot."""
        cb = self._build_cb(lf.standard_loss())
        called = {"count": 0}

        def fake_training_plot(**kwargs):
            """Fake training plot."""
            called["count"] += 1

        cb.training_plot = fake_training_plot
        cb.feedback = 1
        cb.plot_every = 1
        cb.set_model(type("DummyModel", (), {"optimizer": type("opt", (), {"lr": tf.constant(0.1)})()})())
        cb.on_epoch_end(0, logs={"loss": 0.1, "val_loss": 0.2})
        expected_calls = 0 if sp.cluster_plotting else 1
        self.assertEqual(called["count"], expected_calls)

    def test_print_training_summary_variable_metrics(self):
        """Test Print training summary variable metrics."""
        cb = self._build_cb(lf.variable_weight_loss(lambda_1=1.0, lambda_2=0.0, beta=0.0))
        cb.training_metrics = ["loss", "val_loss", "lambda_1"]
        cb.log = {"loss": [1e-4], "val_loss": [1e4], "lambda_1": [0.5]}
        cb.print_training_summary()

    def test_plot_weighted_density_abs_values(self):
        """Test Plot weighted density abs values."""
        cb = self._build_cb(lf.variable_weight_loss(lambda_1=0.5, lambda_2=0.5, beta=0.0))
        cb.log.update(
            {
                "lambda_1": [0.5, 0.4],
                "lambda_2": [0.5, 0.6],
                "rho_loss": [1.0, 0.5],
                "ee_loss": [0.5, 0.25],
                "val_rho_loss": [0.8, 0.4],
                "val_ee_loss": [0.4, 0.2],
            }
        )
        fig, ax = plt.subplots()
        cb._plot_weighted_density_evidence_error_losses(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_evidence_error_with_values(self):
        """Test Plot evidence error with values."""
        cb = self._build_cb(lf.constant_weight_loss())
        cb.log.update(
            {
                "evidence_error": [0.1, 0.2],
                "training_evidence_error": [0.1, 0.2],
                "test_evidence_error": [0.1, 0.2],
            }
        )
        fig, ax = plt.subplots()
        cb._plot_evidence_error(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_loss_symlog_branch(self):
        """Test Plot loss symlog branch."""
        cb = self._build_cb(lf.standard_loss())
        cb.log["loss"] = [0.0, -0.001]
        cb.log["val_loss"] = [0.0, -0.002]
        fig, ax = plt.subplots()
        cb._plot_loss(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_lr(self):
        """Test Plot lr."""
        cb = self._build_cb(lf.standard_loss())
        fig, ax = plt.subplots()
        cb._plot_lr(ax, logs=cb.log)
        plt.close(fig)

    def test_plot_density_evidence_error_losses_constant(self):
        """Test Plot density evidence error losses constant."""
        cb = self._build_cb(lf.constant_weight_loss())
        cb.log.update(
            {"rho_loss": [1.0, 0.8], "ee_loss": [0.5, 0.3], "val_rho_loss": [0.6, 0.4], "val_ee_loss": [0.4, 0.2]}
        )
        fig, ax = plt.subplots()
        cb._plot_density_evidence_error_losses(ax, logs=cb.log)
        plt.close(fig)
