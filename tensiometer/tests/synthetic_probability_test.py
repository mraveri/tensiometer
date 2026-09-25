"""
Tests for the synthetic probability flows of
:mod:`tensiometer.synthetic_probability.synthetic_probability`.

The tests build small flows on small Gaussian chains and train them for a few epochs.
They cover chain initialization, the fixed and trainable bijector options, the training
split, the loss modes, training and monitoring, plotting, sampling, the geometry API,
derived and transformed flows, average flows and the cache helpers. In-depth persistence
and API contract checks live in ``synthetic_probability_persistence_test.py`` and
``synthetic_probability_api_contract_test.py``.
"""

#########################################################################################################
# Imports

import contextlib
import copy
import importlib.util
import inspect
import io
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import matplotlib
matplotlib.use('agg')  # before pyplot, so that no window is opened
from matplotlib import pyplot as plt
import numpy as np
import torch
from getdist import MCSamples
from scipy.spatial import cKDTree

import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import bijectors as bj
from tensiometer.synthetic_probability import loss_functions as lf
from tensiometer.synthetic_probability import tensor_utilities as tu
from tensiometer.synthetic_probability import trainable_bijectors as tb
from tensiometer.synthetic_probability import training

#########################################################################################################
# Helpers

# options of a small flow that trains in a few milliseconds:
SMALL_FLOW = {
    'feedback': 0,
    'plot_every': 0,
    'hidden_units': [8],
    'n_transformations': 1,
    'validation_split': 0.2,
}

# options of a short training run:
SHORT_TRAINING = {
    'epochs': 2,
    'steps_per_epoch': 2,
    'batch_size': 32,
}

GAUSSIAN_MEAN = np.array([0.1, -0.2])
GAUSSIAN_COV = np.array([[0.04, 0.012], [0.012, 0.09]])


def tolerance():
    """
    Absolute tolerance for numerical comparisons at the active precision.

    :returns: float tolerance.
    """
    if tu.get_precision() == torch.float64:
        return 1.e-8
    return 1.e-4


def seed_everything(seed=0):
    """
    Seed numpy and torch.

    :param seed: random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_gaussian_chain(num_samples=500, with_loglikes=True, weights=None, name_tag='gauss', seed=0,
                        ranges=None):
    """
    Two dimensional correlated Gaussian chain.

    :param num_samples: number of samples.
    :param with_loglikes: attach ``-log P`` (up to a constant) to the chain.
    :param weights: optional sample weights.
    :param name_tag: name tag of the chain.
    :param seed: seed of the sample generator.
    :param ranges: optional getdist ranges.
    :returns: :class:`~getdist.mcsamples.MCSamples` with parameters ``p1`` and ``p2``.
    """
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(GAUSSIAN_MEAN, GAUSSIAN_COV, size=num_samples)
    loglikes = None
    if with_loglikes:
        diff = samples - GAUSSIAN_MEAN
        loglikes = 0.5 * np.einsum('ij,jk,ik->i', diff, np.linalg.inv(GAUSSIAN_COV), diff)
    return MCSamples(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        names=['p1', 'p2'],
        labels=['p_1', 'p_2'],
        ranges=ranges,
        name_tag=name_tag,
    )


def make_three_param_chain(num_samples=300, seed=1):
    """
    Three dimensional uncorrelated Gaussian chain with loglikes.

    :param num_samples: number of samples.
    :param seed: seed of the sample generator.
    :returns: :class:`~getdist.mcsamples.MCSamples` with parameters ``a``, ``b``, ``c``.
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(num_samples, 3))
    return MCSamples(
        samples=samples,
        loglikes=0.5 * np.sum(samples**2, axis=1),
        names=['a', 'b', 'c'],
        labels=['a', 'b', 'c'],
        name_tag='three',
    )


def build_flow(chain=None, **kwargs):
    """
    Build a small flow with the options in :data:`SMALL_FLOW` updated by ``kwargs``.

    :param chain: input chain, defaults to :func:`make_gaussian_chain`.
    :param kwargs: options of :class:`~tensiometer.synthetic_probability.synthetic_probability.FlowCallback`.
    :returns: the flow.
    """
    if chain is None:
        chain = make_gaussian_chain()
    options = dict(SMALL_FLOW)
    options.update(kwargs)
    return sp.FlowCallback(chain, **options)


def build_trained_flow(chain=None, **kwargs):
    """
    Build a small flow and train it for a few epochs.

    :param chain: input chain, defaults to :func:`make_gaussian_chain`.
    :param kwargs: options of the flow.
    :returns: the trained flow.
    """
    flow = build_flow(chain, **kwargs)
    flow.train(**SHORT_TRAINING)
    return flow


def quiet():
    """
    Context manager that silences standard output.

    :returns: context manager redirecting ``sys.stdout``.
    """
    return contextlib.redirect_stdout(io.StringIO())


def to_array(value):
    """
    Convert a tensor to a float64 numpy array.

    :param value: tensor or array-like.
    :returns: numpy array.
    """
    return np.asarray(tu.to_numpy(value), dtype=np.float64)

#########################################################################################################
# Test bijectors and transformations (module level so that they pickle)


class TrainableShift(bj.Bijector):
    """Elementwise shift with a trainable parameter, the smallest trainable bijector."""

    min_event_ndims = 0

    def __init__(self, dimension):
        """
        :param dimension: number of parameters.
        """
        super().__init__(name='trainable_shift')
        self.shift = torch.nn.Parameter(torch.zeros(dimension, dtype=tu.prec))
        self.reset_count = 0

    def reset_parameters(self):
        """Draw a new random shift and count the calls."""
        self.reset_count += 1
        with torch.no_grad():
            self.shift.normal_(0.0, 0.1)

    def _forward(self, x):
        """Add the shift."""
        return x + self.shift

    def _inverse(self, y):
        """Remove the shift."""
        return y - self.shift

    def _forward_log_det_jacobian(self, x):
        """Zero log determinant."""
        return torch.zeros_like(x)


class ShiftTransformation(tb.TrainableTransformation):
    """Trainable transformation wrapping :class:`TrainableShift`."""

    def __init__(self, dimension):
        """
        :param dimension: number of parameters.
        """
        self.bijector = TrainableShift(dimension)


def sum_difference_forward(x):
    """Map ``(a, b)`` to ``(a + b, a - b)``."""
    return torch.stack([x[..., 0] + x[..., 1], x[..., 0] - x[..., 1]], dim=-1)


def sum_difference_inverse(y):
    """Map ``(s, d)`` back to ``((s + d) / 2, (s - d) / 2)``."""
    return torch.stack([0.5 * (y[..., 0] + y[..., 1]), 0.5 * (y[..., 0] - y[..., 1])], dim=-1)


class PlotTestCase(unittest.TestCase):
    """Base class closing all figures after each test."""

    def tearDown(self):
        """Close all figures."""
        plt.close('all')

#########################################################################################################
# Module level settings


class TestModuleSettings(unittest.TestCase):
    """Precision forwarding and conversion helpers."""

    def test_prec_forwards_to_tensor_utilities(self):
        """``sp.prec`` and ``sp.np_prec`` are the live tensor_utilities values."""
        self.assertIs(sp.prec, tu.get_precision())
        self.assertIs(sp.np_prec, tu.np_prec)

    def test_unknown_module_attribute_raises(self):
        """Other module attributes raise AttributeError."""
        with self.assertRaises(AttributeError):
            getattr(sp, 'not_an_attribute')

    def test_cast_returns_cpu_tensor(self):
        """``cast`` gives a CPU tensor with the active precision."""
        flow = build_flow(trainable_bijector=None)
        value = flow.cast([1.0, 2.0])
        self.assertTrue(torch.is_tensor(value))
        self.assertEqual(value.dtype, tu.get_precision())
        self.assertEqual(value.device.type, 'cpu')
        self.assertEqual(flow.prec, tu.get_precision())
        self.assertIs(flow.np_prec, tu.np_prec)

    def test_normalize_weights(self):
        """Weights are rescaled to sum to the number of samples."""
        np.testing.assert_allclose(sp._normalize_weights([1.0, 3.0]), [0.5, 1.5])
        self.assertEqual(len(sp._normalize_weights([])), 0)

    def test_degenerate_weights_become_uniform(self):
        """Weights with a non-positive or non-finite sum are replaced by uniform weights."""
        for weights in ([0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [1.0, np.inf, 1.0], [np.nan, 1.0, 1.0]):
            np.testing.assert_array_equal(sp._normalize_weights(weights), np.ones(3))


class TestModuleImport(unittest.TestCase):
    """Import-time plotting settings, checked on fresh copies of the module."""

    def load_fresh_module(self):
        """
        Execute the module source in a new module object that is not registered in ``sys.modules``.

        :returns: the new module.
        """
        spec = importlib.util.spec_from_file_location(sp.__name__, sp.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_import_without_ipython(self):
        """Without IPython the module imports and has no ``clear_output``."""
        saved = sys.modules.get('IPython.display', None)
        sys.modules['IPython.display'] = None
        try:
            module = self.load_fresh_module()
        finally:
            if saved is None:
                sys.modules.pop('IPython.display', None)
            else:
                sys.modules['IPython.display'] = saved
        self.assertFalse(hasattr(module, 'clear_output'))
        self.assertTrue(issubclass(module.FlowCallback, training.Callback))

    def test_interactive_backend_turns_on_interactive_mode(self):
        """With an interactive backend (neither inline nor agg) pyplot interactive mode is switched on."""
        with patch.object(matplotlib, 'get_backend', return_value='MacOSX'), patch.object(plt, 'ion') as ion:
            module = self.load_fresh_module()
        ion.assert_called_once_with()
        self.assertFalse(module.ipython_plotting)
        self.assertFalse(module.cluster_plotting)

    def test_agg_backend_keeps_interactive_mode_off(self):
        """With the agg backend plots are not shown and interactive mode is left alone."""
        with patch.object(matplotlib, 'get_backend', return_value='agg'), patch.object(plt, 'ion') as ion:
            module = self.load_fresh_module()
        ion.assert_not_called()
        self.assertTrue(module.cluster_plotting)

#########################################################################################################
# Chain initialization and input validation


class TestChainInitialization(unittest.TestCase):
    """Reading the chain into the flow."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_basic_attributes(self):
        """Names, labels, sizes and dtypes are read from the chain."""
        chain = make_gaussian_chain()
        flow = build_flow(chain, trainable_bijector=None)
        self.assertEqual(flow.num_params, 2)
        self.assertEqual(flow.param_names, ['p1', 'p2'])
        self.assertEqual(flow.param_labels, ['p_1', 'p_2'])
        self.assertEqual(flow.name_tag, 'gauss_flow')
        self.assertEqual(flow.chain_samples.shape, (500, 2))
        self.assertEqual(flow.chain_samples.dtype, tu.np_prec)
        self.assertEqual(flow.chain_weights.dtype, tu.np_prec)
        self.assertTrue(flow.has_loglikes)
        self.assertEqual(flow.chain_loglikes.dtype, tu.np_prec)
        self.assertFalse(flow.is_trained)
        self.assertFalse(flow.is_light)
        self.assertIsNone(flow.MAP_coord)
        self.assertIsNone(flow.MAP_logP)
        self.assertEqual(flow.device, tu.check_device(None))

    def test_name_tag_defaults_to_flow(self):
        """Chains without a name tag give the name ``flow``."""
        chain = make_gaussian_chain(name_tag=None)
        flow = build_flow(chain, trainable_bijector=None)
        self.assertEqual(flow.name_tag, 'flow')

    def test_sample_map_from_loglikes_and_weights(self):
        """The sample MAP is the best loglike, or the largest weight without loglikes."""
        chain = make_gaussian_chain()
        flow = build_flow(chain, trainable_bijector=None)
        best = chain.samples[np.argmin(chain.loglikes)]
        np.testing.assert_allclose(flow.sample_MAP, best)
        weights = np.ones(500)
        weights[7] = 5.0
        chain = make_gaussian_chain(with_loglikes=False, weights=weights)
        flow = build_flow(chain, trainable_bijector=None)
        self.assertFalse(flow.has_loglikes)
        self.assertIsNone(flow.chain_loglikes)
        np.testing.assert_allclose(flow.sample_MAP, chain.samples[7])

    def test_param_names_subset(self):
        """Only the requested parameters are used, in the requested order."""
        flow = build_flow(make_three_param_chain(), param_names=['c', 'a'], trainable_bijector=None)
        self.assertEqual(flow.num_params, 2)
        self.assertEqual(flow.param_names, ['c', 'a'])
        chain = make_three_param_chain()
        np.testing.assert_allclose(flow.chain_samples, chain.samples[:, [2, 0]].astype(tu.np_prec))

    def test_missing_param_name_raises(self):
        """Unknown parameter names raise ValueError."""
        with self.assertRaises(ValueError):
            build_flow(param_names=['p1', 'missing'], trainable_bijector=None)

    def test_missing_periodic_param_raises(self):
        """Unknown periodic parameters raise ValueError."""
        with self.assertRaises(ValueError):
            build_flow(periodic_params=['missing'], trainable_bijector=None)

    def test_periodic_param_as_string(self):
        """A single periodic parameter can be given as a string."""
        chain = make_gaussian_chain(ranges={'p1': [-1.0, 1.0]})
        flow = build_flow(chain, periodic_params='p1', trainable_bijector=None)
        self.assertEqual(flow.periodic_params, ['p1'])

    def test_custom_param_ranges(self):
        """Explicit ranges are stored."""
        ranges = {'p1': [-2.0, 2.0], 'p2': [-3.0, 3.0]}
        flow = build_flow(param_ranges=ranges, trainable_bijector=None)
        self.assertEqual(flow.parameter_ranges, ranges)
        self.assertIsNot(flow.parameter_ranges['p1'], ranges['p1'])

    def test_ranges_from_chain(self):
        """Ranges come from the getdist ranges or, if missing, from the sample extent."""
        chain = make_gaussian_chain(ranges={'p1': [-1.0, None]})
        flow = build_flow(chain, trainable_bijector=None)
        self.assertEqual(flow.parameter_ranges['p1'][0], -1.0)
        self.assertAlmostEqual(flow.parameter_ranges['p1'][1], np.amax(chain.samples[:, 0]))
        self.assertAlmostEqual(flow.parameter_ranges['p2'][0], np.amin(chain.samples[:, 1]))
        self.assertAlmostEqual(flow.parameter_ranges['p2'][1], np.amax(chain.samples[:, 1]))

    def test_incomplete_param_ranges_raise(self):
        """All parameters need a range when ranges are given."""
        with self.assertRaises(ValueError):
            build_flow(param_ranges={'p1': [-2.0, 2.0]}, trainable_bijector=None)

    def test_samples_outside_ranges_raise(self):
        """Samples outside the given ranges raise ValueError."""
        with self.assertRaises(ValueError):
            build_flow(param_ranges={'p1': [0.0, 0.05], 'p2': [-3.0, 3.0]}, trainable_bijector=None)

    def test_feedback_and_plot_every_validation(self):
        """Negative or non integer feedback and plot_every raise ValueError."""
        with self.assertRaises(ValueError):
            build_flow(feedback=-1, trainable_bijector=None)
        with self.assertRaises(ValueError):
            build_flow(feedback=1.5, trainable_bijector=None)
        with self.assertRaises(ValueError):
            build_flow(plot_every=-1, trainable_bijector=None)

    def test_feedback_none_is_zero(self):
        """``feedback=None`` means no feedback."""
        flow = build_flow(feedback=None, trainable_bijector=None)
        self.assertEqual(flow.feedback, 0)

    def test_trainable_bijector_path_raises(self):
        """The removed ``trainable_bijector_path`` option raises ValueError."""
        with self.assertRaises(ValueError):
            build_flow(trainable_bijector_path='some/path')

    def test_nearest_samples(self):
        """``init_nearest`` builds the nearest neighbour index."""
        flow = build_flow(init_nearest=True, trainable_bijector=None)
        self.assertEqual(flow.chain_nearest_index.shape, (500, 2))
        np.testing.assert_array_equal(flow.chain_nearest_index[:, 0], np.arange(500))
        flow = build_flow(trainable_bijector=None)
        self.assertNotIn('chain_nearest_index', flow.__dict__)
        flow._init_nearest_samples()
        self.assertEqual(flow.chain_nearest_index.shape, (500, 2))

    def test_verbose_feedback(self):
        """High feedback levels print the initialization steps."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            flow = build_flow(feedback=2, loss_mode='fixed')
        text = output.getvalue()
        self.assertIn('Initializing samples', text)
        self.assertIn('Initializing trainable bijector', text)
        self.assertIn('Initializing trainer', text)
        self.assertEqual(flow.feedback, 2)

    def test_verbose_feedback_without_rescaling(self):
        """High feedback reports that the samples are not rescaled."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            build_flow(feedback=2, apply_rescaling=False, trainable_bijector=None)
        self.assertIn('not rescaling samples', output.getvalue())

    def test_verbose_feedback_weighted_chain(self):
        """High feedback reports the effective number of samples of weighted chains."""
        weights = np.random.default_rng(3).uniform(0.5, 1.5, size=500)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            flow = build_flow(make_gaussian_chain(weights=weights), feedback=2, trainable_bijector=None)
        text = output.getvalue()
        self.assertTrue(flow.has_weights)
        self.assertIn('non-uniform weights', text)
        training_ess = np.sum(flow.training_weights)**2 / np.sum(flow.training_weights**2)
        test_ess = np.sum(flow.test_weights)**2 / np.sum(flow.test_weights**2)
        self.assertIn('{0:.6g} effective number of training samples'.format(training_ess), text)
        self.assertIn('{0:.6g} effective number of test samples'.format(test_ess), text)

    def test_init_chain_without_chain(self):
        """``_init_chain`` without a chain leaves the flow unchanged."""
        flow = build_flow(trainable_bijector=None)
        samples = flow.chain_samples
        self.assertIsNone(flow._init_chain(None))
        self.assertIs(flow.chain_samples, samples)
        self.assertEqual(flow.param_names, ['p1', 'p2'])

    def test_nearest_samples_single_sample(self):
        """With one sample no covariance is estimated and the neighbour is flagged as missing."""
        flow = build_flow(trainable_bijector=None)
        flow.chain_samples = flow.chain_samples[:1]
        flow.chain_weights = flow.chain_weights[:1]
        flow._init_nearest_samples()
        self.assertEqual(flow.chain_nearest_index.shape, (1, 2))
        self.assertEqual(flow.chain_nearest_index[0, 0], 0)
        # cKDTree marks missing neighbours with the number of points:
        self.assertEqual(flow.chain_nearest_index[0, 1], 1)

    def test_nearest_samples_degenerate_weight_sum(self):
        """Weights with a non-positive sum fall back to the unweighted covariance."""
        flow = build_flow(trainable_bijector=None)
        flow._init_nearest_samples()
        uniform_index = flow.chain_nearest_index.copy()
        flow.chain_weights = np.zeros_like(flow.chain_weights)
        flow._init_nearest_samples()
        np.testing.assert_array_equal(flow.chain_nearest_index, uniform_index)

    def test_nearest_samples_non_finite_covariance(self):
        """A non-finite weighted covariance falls back to Euclidean nearest neighbours."""
        flow = build_flow(trainable_bijector=None)
        weights = np.zeros_like(flow.chain_weights)
        weights[0] = 1.0
        flow.chain_weights = weights
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flow._init_nearest_samples()
        _, expected = cKDTree(flow.chain_samples).query(flow.chain_samples, 2)
        np.testing.assert_array_equal(flow.chain_nearest_index, expected)

#########################################################################################################
# Fixed bijector: prior and rescaling options


class TestFixedBijector(unittest.TestCase):
    """Prior bijector, whitening and periodic parameters."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_prior_ranges(self):
        """The default prior maps the parameter ranges to an unbounded space."""
        flow = build_flow(trainable_bijector=None)
        self.assertIsInstance(flow.prior_bijector, bj.Blockwise)
        self.assertIs(flow.bijectors[0], flow.prior_bijector)
        with torch.no_grad():
            abstract = flow.prior_bijector.inverse(flow.chain_samples)
        self.assertTrue(bool(torch.isfinite(abstract).all()))
        with torch.no_grad():
            back = flow.prior_bijector.forward(abstract)
        np.testing.assert_allclose(to_array(back), flow.chain_samples, atol=100 * tolerance())

    def test_prior_bijector_object(self):
        """A bijector passed as prior is used as is."""
        prior = bj.Shift(0.5)
        flow = build_flow(prior_bijector=prior, trainable_bijector=None)
        self.assertIs(flow.prior_bijector, prior)

    def test_prior_none_or_false(self):
        """None and False give the identity prior."""
        for option in [None, False]:
            flow = build_flow(prior_bijector=option, trainable_bijector=None)
            self.assertIsInstance(flow.prior_bijector, bj.Identity)

    def test_invalid_prior_raises(self):
        """Unknown prior options raise ValueError."""
        with self.assertRaises(ValueError):
            build_flow(prior_bijector='uniform', trainable_bijector=None)

    def test_rescaling_true_whitens(self):
        """Full rescaling whitens the training samples."""
        flow = build_flow(prior_bijector=None, trainable_bijector=None, validation_split=0.0)
        affine = [_b for _b in flow.bijectors if isinstance(_b, bj.AffineTriL)]
        self.assertEqual(len(affine), 1)
        self.assertNotEqual(float(affine[0].scale_tril[1, 0]), 0.0)
        samples = flow.training_samples
        np.testing.assert_allclose(np.mean(samples, axis=0), np.zeros(2), atol=1.e-4)
        np.testing.assert_allclose(np.cov(samples.T), np.eye(2), atol=1.e-2)

    def test_rescaling_independent(self):
        """Independent rescaling uses a diagonal scale."""
        flow = build_flow(prior_bijector=None, trainable_bijector=None, apply_rescaling='independent',
                          validation_split=0.0)
        affine = [_b for _b in flow.bijectors if isinstance(_b, bj.AffineTriL)]
        self.assertEqual(len(affine), 1)
        scale = to_array(affine[0].scale_tril)
        np.testing.assert_allclose(scale, np.diag(np.diagonal(scale)))
        np.testing.assert_allclose(np.var(flow.training_samples, axis=0), np.ones(2), atol=1.e-2)
        self.assertGreater(abs(np.corrcoef(flow.training_samples.T)[0, 1]), 0.1)

    def test_rescaling_false(self):
        """Without rescaling there is no affine bijector."""
        flow = build_flow(prior_bijector=None, trainable_bijector=None, apply_rescaling=False,
                          validation_split=0.0)
        self.assertFalse(any(isinstance(_b, bj.AffineTriL) for _b in flow.bijectors))
        np.testing.assert_allclose(np.sort(flow.training_samples, axis=0),
                                   np.sort(flow.chain_samples, axis=0), atol=tolerance())

    def test_periodic_requires_rescaling(self):
        """Periodic parameters need rescaling."""
        chain = make_gaussian_chain(ranges={'p1': [-1.0, 1.0]})
        with self.assertRaises(ValueError):
            build_flow(chain, periodic_params=['p1'], apply_rescaling=False, trainable_bijector=None)

    def test_periodic_wide_parameter_is_trainable(self):
        """A periodic parameter spread over its period is left to the trainable bijector."""
        chain = make_gaussian_chain(ranges={'p1': [-1.0, 1.0]})
        flow = build_flow(chain, periodic_params=['p1'], trainable_bijector=None)
        names = [_b.name for _b in flow.bijectors]
        self.assertIn('ModBijector', names)
        self.assertEqual(flow.trainable_periodic_params, ['p1'])
        self.assertIsInstance(flow.prior_bijector.bijectors[0], bj.Identity)

    def test_periodic_localized_parameter_is_rescaled(self):
        """A periodic parameter well localized in its period is rescaled, not trained."""
        chain = make_gaussian_chain(ranges={'p1': [-10.0, 10.0]})
        flow = build_flow(chain, periodic_params=['p1'], trainable_bijector=None)
        self.assertEqual(flow.trainable_periodic_params, [])
        mod_bijector = [_b for _b in flow.bijectors if _b.name == 'ModBijector'][0]
        self.assertIsInstance(mod_bijector.bijectors[0].bijectors[-1], bj.Scale)
        self.assertEqual(mod_bijector.bijectors[1].name, 'identity')

    def test_periodic_spline_flow_trains(self):
        """A flow with a trainable periodic parameter builds and trains with splines."""
        chain = make_gaussian_chain(ranges={'p1': [-1.0, 1.0]})
        flow = build_flow(chain, periodic_params=['p1'], transformation_type='spline')
        history = flow.train(**SHORT_TRAINING)
        self.assertTrue(np.all(np.isfinite(history.history['loss'])))

#########################################################################################################
# Trainable bijector options


class TestTrainableBijectorOptions(unittest.TestCase):
    """Choices for the trainable part of the flow."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_autoregressive_flow(self):
        """The default builds an AutoregressiveFlow."""
        flow = build_flow()
        self.assertIsInstance(flow.trainable_transformation, tb.AutoregressiveFlow)
        self.assertIs(flow.trainable_bijector, flow.trainable_transformation.bijector)
        self.assertGreater(training.count_parameters(flow.trainable_bijector), 0)

    def test_trainable_transformation_object(self):
        """A TrainableTransformation is used through its bijector."""
        transformation = ShiftTransformation(2)
        flow = build_flow(trainable_bijector=transformation)
        self.assertIs(flow.trainable_transformation, transformation)
        self.assertIs(flow.trainable_bijector, transformation.bijector)

    def test_bijector_object(self):
        """A bijector is used directly, without a trainable transformation."""
        shift = TrainableShift(2)
        flow = build_flow(trainable_bijector=shift)
        self.assertIsNone(flow.trainable_transformation)
        self.assertIs(flow.trainable_bijector, shift)

    def test_none_and_false(self):
        """None and False give the identity."""
        for option in [None, False]:
            flow = build_flow(trainable_bijector=option)
            self.assertIsNone(flow.trainable_transformation)
            self.assertIsInstance(flow.trainable_bijector, bj.Identity)

    def test_invalid_option_raises(self):
        """Unknown options raise ValueError."""
        with self.assertRaises(ValueError):
            build_flow(trainable_bijector='unknown')

    def test_bijector_chain_and_distributions(self):
        """The full bijector chains fixed and trainable parts, used by the distributions."""
        flow = build_flow()
        self.assertIsInstance(flow.bijector, bj.Chain)
        self.assertIs(flow.bijector.bijectors[-1], flow.trainable_bijector)
        self.assertIs(flow.bijectors[-1], flow.trainable_bijector)
        self.assertIs(flow.distribution.bijector, flow.bijector)
        self.assertIs(flow.trained_distribution.bijector, flow.trainable_bijector)
        samples = flow.distribution.sample(5)
        self.assertEqual(tuple(samples.shape), (5, 2))

    def test_bijectors_on_flow_device(self):
        """Trainable parameters live on the flow device."""
        flow = build_flow()
        for parameter in flow.trainable_bijector.parameters():
            self.assertEqual(parameter.device, flow.device)
            self.assertEqual(parameter.dtype, tu.get_precision())

#########################################################################################################
# Training dataset and split


class TestTrainingDataset(unittest.TestCase):
    """Training and validation split."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_split_sizes_and_disjoint(self):
        """The split has the requested size, is disjoint and covers the chain."""
        flow = build_flow(validation_split=0.25, trainable_bijector=None)
        self.assertEqual(flow.num_test_samples, 125)
        self.assertEqual(flow.num_training_samples, 375)
        self.assertEqual(len(np.intersect1d(flow.test_idx, flow.training_idx)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([flow.test_idx, flow.training_idx])), np.arange(500))
        self.assertEqual(flow.training_samples.shape, (375, 2))
        self.assertEqual(flow.test_samples.shape, (125, 2))
        self.assertEqual(flow.training_samples.dtype, tu.np_prec)

    def test_rng_is_reproducible(self):
        """The same generator seed gives the same split."""
        flow_1 = build_flow(rng=np.random.default_rng(3), trainable_bijector=None)
        flow_2 = build_flow(rng=np.random.default_rng(3), trainable_bijector=None)
        flow_3 = build_flow(rng=np.random.default_rng(4), trainable_bijector=None)
        np.testing.assert_array_equal(flow_1.test_idx, flow_2.test_idx)
        np.testing.assert_array_equal(flow_1.training_idx, flow_2.training_idx)
        self.assertFalse(np.array_equal(flow_1.test_idx, flow_3.test_idx))

    def test_validation_training_idx(self):
        """An explicit split is used as given."""
        test_idx = np.arange(0, 100)
        training_idx = np.arange(100, 500)
        flow = build_flow(validation_training_idx=(test_idx, training_idx), trainable_bijector=None)
        np.testing.assert_array_equal(flow.test_idx, test_idx)
        np.testing.assert_array_equal(flow.training_idx, training_idx)
        flow._init_training_dataset(validation_training_idx=(training_idx[:10], test_idx))
        self.assertEqual(flow.num_test_samples, 10)
        self.assertEqual(flow.num_training_samples, 100)

    def test_dataset_tensors_with_loglikes(self):
        """The datasets are ``(x, logP, w)`` tensors on the flow device."""
        flow = build_flow(trainable_bijector=None)
        for dataset, size in [(flow.training_dataset, 400), (flow.validation_dataset, 100)]:
            self.assertEqual(len(dataset), 3)
            x, logP, weights = dataset
            for tensor in (x, logP, weights):
                self.assertEqual(tensor.dtype, tu.get_precision())
                self.assertEqual(tensor.device, flow.device)
            self.assertEqual(tuple(x.shape), (size, 2))
            self.assertEqual(tuple(logP.shape), (size,))
            self.assertEqual(tuple(weights.shape), (size,))

    def test_dataset_tensors_without_loglikes(self):
        """Without loglikes the target is None and the weights are kept."""
        flow = build_flow(make_gaussian_chain(with_loglikes=False), trainable_bijector=None)
        self.assertIsNone(flow.training_logP_preabs)
        self.assertIsNone(flow.test_logP_preabs)
        self.assertIsNone(flow.training_dataset[1])
        self.assertIsNone(flow.validation_dataset[1])
        self.assertEqual(tuple(flow.training_dataset[2].shape), (400,))

    def test_logP_preabs(self):
        """The target log posterior includes the Jacobian of the fixed bijector."""
        flow = build_flow(trainable_bijector=None)
        chain_samples = flow.chain_samples[flow.training_idx]
        with torch.no_grad():
            log_det = to_array(flow.fixed_bijector.inverse_log_det_jacobian(chain_samples, event_ndims=1))
        expected = -flow.chain_loglikes[flow.training_idx] - log_det
        np.testing.assert_allclose(flow.training_logP_preabs, expected, rtol=10 * tolerance(), atol=10 * tolerance())

    def test_weights_normalization(self):
        """Weights are normalized to the number of samples; non-uniform weights are flagged."""
        flow = build_flow(trainable_bijector=None)
        self.assertFalse(flow.has_weights)
        weights = np.random.default_rng(0).uniform(0.5, 2.0, size=500)
        flow = build_flow(make_gaussian_chain(with_loglikes=False, weights=weights), trainable_bijector=None)
        self.assertTrue(flow.has_weights)
        self.assertAlmostEqual(np.sum(flow.training_weights), flow.num_training_samples, places=4)
        self.assertAlmostEqual(np.sum(flow.test_weights), flow.num_test_samples, places=4)
        ratio = flow.training_weights / weights[flow.training_idx]
        np.testing.assert_allclose(ratio, ratio[0] * np.ones_like(ratio), rtol=1.e-5)

#########################################################################################################
# Loss modes


class TestLossModes(unittest.TestCase):
    """Loss function options."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_loss_classes(self):
        """Each loss mode selects its loss class."""
        expected = {
            'standard': lf.standard_loss,
            'fixed': lf.constant_weight_loss,
            'random': lf.random_weight_loss,
            'annealed': lf.annealed_weight_loss,
            'softadapt': lf.SoftAdapt_weight_loss,
            'sharpstep': lf.SharpStep,
        }
        flow = build_flow(trainable_bijector=None)
        self.assertEqual(flow.loss_mode, 'standard')
        for mode, loss_class in expected.items():
            flow._init_loss_function(loss_mode=mode, learning_rate=1.e-4)
            self.assertIsInstance(flow.loss, loss_class)
            self.assertEqual(flow.loss_mode, mode)
            self.assertEqual(flow.initial_learning_rate, 1.e-4)
            self.assertAlmostEqual(flow.final_learning_rate, 1.e-7)

    def test_fixed_loss_parameters(self):
        """The fixed loss uses alpha_lossv and beta_lossv."""
        flow = build_flow(trainable_bijector=None, loss_mode='fixed', alpha_lossv=0.3, beta_lossv=0.1)
        self.assertEqual(flow.alpha_lossv, 0.3)
        self.assertEqual(flow.beta_lossv, 0.1)

    def test_posterior_losses_require_loglikes(self):
        """Losses using the posterior need loglikes."""
        with self.assertRaises(ValueError):
            build_flow(make_gaussian_chain(with_loglikes=False), trainable_bijector=None, loss_mode='fixed')

    def test_unknown_loss_mode_raises(self):
        """Unknown loss modes raise ValueError."""
        with self.assertRaises(ValueError):
            build_flow(trainable_bijector=None, loss_mode='unknown')

    def test_training_metrics_per_loss(self):
        """The monitored metrics depend on the loss class."""
        flow = build_flow(trainable_bijector=None)
        self.assertEqual(flow.training_metrics[:2], ['loss', 'val_loss'])
        self.assertNotIn('rho_loss', flow.training_metrics)
        flow = build_flow(trainable_bijector=None, loss_mode='fixed')
        self.assertIn('rho_loss', flow.training_metrics)
        self.assertIn('evidence', flow.training_metrics)
        self.assertNotIn('lambda_1', flow.training_metrics)
        flow = build_flow(trainable_bijector=None, loss_mode='random')
        self.assertIn('lambda_1', flow.training_metrics)
        self.assertIn('lambda_2', flow.training_metrics)
        self.assertEqual(set(flow.log.keys()), set(flow.training_metrics))

    def test_training_with_each_loss_mode(self):
        """Every loss mode trains and fills its logs."""
        for mode in ['fixed', 'random', 'annealed', 'softadapt', 'sharpstep']:
            flow = build_flow(loss_mode=mode)
            history = flow.train(**SHORT_TRAINING)
            self.assertTrue(np.all(np.isfinite(history.history['loss'])), mode)
            for key in flow.training_metrics:
                self.assertEqual(len(flow.log[key]), SHORT_TRAINING['epochs'], mode + ' ' + key)
            if mode != 'fixed':
                self.assertTrue(np.all(np.isfinite(flow.log['lambda_1'])))

#########################################################################################################
# Training


class TestTraining(unittest.TestCase):
    """Trainer creation, train and global_train."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_trainer_created_lazily(self):
        """With ``initialize_model=False`` the trainer is built by train."""
        flow = build_flow(initialize_model=False)
        self.assertFalse(flow._trainer_initialized)
        self.assertNotIn('trainer', flow.__dict__)
        flow.train(**SHORT_TRAINING)
        self.assertTrue(flow._trainer_initialized)
        self.assertIsInstance(flow.trainer, training.Trainer)

    def test_init_trainer(self):
        """The trainer optimizes the trainable bijector with the flow loss."""
        flow = build_flow()
        self.assertTrue(flow._trainer_initialized)
        self.assertIs(flow.trainer.module, flow.trainable_bijector)
        self.assertIs(flow.trainer.loss, flow.loss)
        self.assertIsInstance(flow.trainer.optimizer, torch.optim.Adam)
        self.assertEqual(flow.trainer.get_learning_rate(), flow.initial_learning_rate)
        self.assertEqual(flow.trainer.global_clipnorm, flow.global_clipnorm)

    def test_reset_optimizer(self):
        """``_reset_optimizer`` resets the loss and creates a new optimizer."""
        flow = build_flow()
        old_optimizer = flow.trainer.optimizer
        with patch.object(flow.loss, 'reset') as reset:
            flow._reset_optimizer()
        reset.assert_called_once_with()
        self.assertIsNot(flow.trainer.optimizer, old_optimizer)

    def test_more_parameters_than_data_warning(self):
        """A warning is printed when the network has more parameters than data."""
        chain = make_gaussian_chain(num_samples=20)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            build_flow(chain, hidden_units=[32, 32])
        self.assertIn('more parameters than data', output.getvalue())

    def test_train_history_and_logs(self):
        """train returns a History and fills the logs once per epoch."""
        flow = build_flow()
        history = flow.train(epochs=3, steps_per_epoch=2, batch_size=32, verbose=0)
        self.assertIsInstance(history, training.History)
        for key in ['loss', 'val_loss', 'lr']:
            self.assertEqual(len(history.history[key]), 3)
        self.assertTrue(flow.is_trained)
        for key in flow.training_metrics:
            self.assertEqual(len(flow.log[key]), 3)
        self.assertEqual(flow.log['loss'], history.history['loss'])
        self.assertEqual(flow.log['val_loss'], history.history['val_loss'])
        self.assertEqual(flow.log['loss_rate'][0], 0.0)
        self.assertAlmostEqual(flow.log['loss_rate'][1], flow.log['loss'][1] - flow.log['loss'][0])
        for value in flow.log['chi2Z_ks_p']:
            self.assertTrue(0.0 <= value <= 1.0)

    def test_training_changes_parameters(self):
        """Training moves the trainable parameters."""
        flow = build_flow()
        before = [_p.detach().clone() for _p in flow.trainable_bijector.parameters()]
        flow.train(**SHORT_TRAINING)
        after = list(flow.trainable_bijector.parameters())
        self.assertTrue(any(not torch.equal(_b, _a.detach()) for _b, _a in zip(before, after)))

    def test_training_reduces_loss(self):
        """A few more epochs reduce the validation loss of a simple trainable bijector."""
        flow = build_flow(trainable_bijector=TrainableShift(2), prior_bijector=None, learning_rate=1.e-2)
        with torch.no_grad():
            flow.trainable_bijector.shift.fill_(1.0)
        history = flow.train(epochs=20, steps_per_epoch=5, batch_size=64, lr_scheduler=None)
        self.assertLess(history.history['val_loss'][-1], history.history['val_loss'][0])

    def test_verbose_levels(self):
        """Explicit verbose levels 0, 1 and -1 run; the default follows feedback."""
        flow = build_flow()
        for verbose in [0, 1, -1]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                flow.train(verbose=verbose, **SHORT_TRAINING)
            if verbose == 1:
                self.assertIn('Epoch 1/2', output.getvalue())
            if verbose == 0:
                self.assertEqual(output.getvalue(), '')

    def test_default_batch_size(self):
        """Without batch size the training sample is divided in steps_per_epoch batches."""
        flow = build_flow()
        with patch.object(flow.trainer, 'fit', wraps=flow.trainer.fit) as fit:
            flow.train(epochs=1)
        self.assertEqual(fit.call_args.kwargs['steps_per_epoch'], 20)
        self.assertEqual(fit.call_args.kwargs['batch_size'], 20)
        with patch.object(flow.trainer, 'fit', wraps=flow.trainer.fit) as fit:
            flow.train(epochs=1, batch_size=100)
        self.assertEqual(fit.call_args.kwargs['steps_per_epoch'], 4)

    def test_lr_scheduler_options(self):
        """The learning rate scheduler is chosen by name, disabled with None."""
        flow = build_flow()
        with patch.object(flow.trainer, 'fit', wraps=flow.trainer.fit) as fit:
            flow.train(lr_scheduler='ExponentialDecayScheduler', lr_max=1.e-3, lr_min=1.e-5, roll_off_step=1,
                       steps=4, **SHORT_TRAINING)
        callbacks = fit.call_args.kwargs['callbacks']
        self.assertIs(callbacks[0], flow)
        self.assertEqual(type(callbacks[1]).__name__, 'ExponentialDecayScheduler')
        with patch.object(flow.trainer, 'fit', wraps=flow.trainer.fit) as fit:
            flow.train(lr_scheduler=None, **SHORT_TRAINING)
        self.assertEqual(fit.call_args.kwargs['callbacks'], [flow])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            flow.train(lr_scheduler='NotAScheduler', **SHORT_TRAINING)
        self.assertIn('not found', output.getvalue())

    def test_custom_callbacks(self):
        """User callbacks replace the scheduler and receive the epoch hooks."""

        class EpochCounter(training.Callback):
            """Count epochs."""

            def __init__(self):
                self.epochs = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epochs += 1

        counter = EpochCounter()
        flow = build_flow()
        flow.train(callbacks=[counter], **SHORT_TRAINING)
        self.assertEqual(counter.epochs, SHORT_TRAINING['epochs'])
        self.assertIs(flow.trainer, counter.trainer)

    def test_unknown_training_kwargs_are_ignored(self):
        """Options of other functions (pop_size, feedback) are filtered out."""
        flow = build_flow()
        history = flow.train(pop_size=3, feedback=0, **SHORT_TRAINING)
        self.assertEqual(len(history.history['loss']), SHORT_TRAINING['epochs'])

    def test_no_trainable_parameters_raises(self):
        """Flows without trainable parameters cannot be trained."""
        flow = build_flow(trainable_bijector=None)
        self.assertIsNone(flow.trainer.optimizer)
        with self.assertRaises(ValueError):
            flow.train(**SHORT_TRAINING)

    def test_weighted_chain_without_loglikes_trains(self):
        """Weights are used as sample weights when there are no loglikes."""
        weights = np.random.default_rng(0).uniform(0.1, 3.0, size=500)
        chain = make_gaussian_chain(with_loglikes=False, weights=weights)
        flow = build_flow(chain)
        with patch.object(flow.trainer, 'fit', wraps=flow.trainer.fit) as fit:
            flow.train(**SHORT_TRAINING)
        np.testing.assert_allclose(to_array(fit.call_args.kwargs['sample_weight']), flow.training_weights,
                                   rtol=tolerance())
        self.assertIsNone(fit.call_args.kwargs['y'])
        self.assertTrue(np.all(np.isfinite(flow.log['loss'])))

    def test_global_train_population(self):
        """global_train keeps the best member of the population."""
        flow = build_flow()
        best_loss, best_val_loss = flow.global_train(pop_size=2, **SHORT_TRAINING)
        self.assertEqual(len(flow.population_logs), 2)
        self.assertEqual([_l['population'] for _l in flow.population_logs], [1, 2])
        final_val_losses = [_l['val_loss'][-1] for _l in flow.population_logs]
        self.assertEqual(best_val_loss, min(final_val_losses))
        best_index = int(np.argmin(final_val_losses))
        self.assertEqual(flow.log['val_loss'], flow.population_logs[best_index]['val_loss'])
        self.assertEqual(best_loss, flow.population_logs[best_index]['loss'][-1])
        self.assertIn('best_loss', flow.population_logs[1])
        self.assertTrue(flow.is_trained)

    def test_global_train_member_without_epochs(self):
        """A member whose training logged no epoch has infinite loss and is not selected."""
        flow = build_flow()
        original_train = sp.FlowCallback.train
        calls = {'count': 0}

        def train_first_fails(self, **kwargs):
            """Return an empty history for the first member."""
            calls['count'] += 1
            if calls['count'] == 1:
                return training.History()
            return original_train(self, **kwargs)

        with patch.object(sp.FlowCallback, 'train', train_first_fails):
            best_loss, best_val_loss = flow.global_train(pop_size=2, **SHORT_TRAINING)
        self.assertTrue(np.isfinite(best_loss))
        self.assertTrue(np.isfinite(best_val_loss))
        self.assertEqual(best_val_loss, flow.population_logs[1]['val_loss'][-1])

    def test_logs_defaults_not_shared(self):
        """Methods taking ``logs`` never default to a shared dictionary."""
        for name in dir(sp.FlowCallback):
            method = getattr(sp.FlowCallback, name)
            if not callable(method):
                continue
            try:
                parameters = inspect.signature(method).parameters
            except (TypeError, ValueError):
                continue
            if 'logs' in parameters:
                self.assertNotIsInstance(parameters['logs'].default, dict, name)
        flow = build_flow()
        _num_losses = len(flow.log['loss'])
        flow.compute_training_metrics()
        self.assertEqual(len(flow.log['loss']), _num_losses + 1)

    def test_global_train_reinitializes_members(self):
        """Population members after the first start from fresh parameters."""
        shift = TrainableShift(2)
        flow = build_flow(trainable_bijector=shift)
        flow.global_train(pop_size=3, **SHORT_TRAINING)
        self.assertEqual(shift.reset_count, 2)

    def test_global_train_restores_best_weights(self):
        """The weights of the best member are restored at the end."""
        flow = build_flow()
        states = []
        original_train = sp.FlowCallback.train

        def recording_train(self, **kwargs):
            """Train and record the weights."""
            history = original_train(self, **kwargs)
            states.append(copy.deepcopy(self.trainable_bijector.state_dict()))
            return history

        with patch.object(sp.FlowCallback, 'train', recording_train):
            flow.global_train(pop_size=2, **SHORT_TRAINING)
        best_index = int(np.argmin([_l['val_loss'][-1] for _l in flow.population_logs]))
        for key, value in flow.trainable_bijector.state_dict().items():
            self.assertTrue(torch.equal(value, states[best_index][key]), key)

    def test_global_train_feedback(self):
        """global_train prints the population summary at high feedback."""
        flow = build_flow()
        flow.feedback = 2
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            flow.global_train(pop_size=2, verbose=0, **SHORT_TRAINING)
        self.assertIn('Training population 2', output.getvalue())
        self.assertIn('best model is number', output.getvalue())

    def test_global_train_builds_trainer(self):
        """global_train builds the trainer of flows created without it and reports single runs."""
        flow = build_flow(initialize_model=False)
        self.assertFalse(flow._trainer_initialized)
        flow.feedback = 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            flow.global_train(pop_size=1, verbose=0, **SHORT_TRAINING)
        self.assertTrue(flow._trainer_initialized)
        self.assertTrue(flow.is_trained)
        self.assertEqual(len(flow.log['loss']), SHORT_TRAINING['epochs'])
        self.assertIn('* Training\n', output.getvalue())
        self.assertNotIn('Training population', output.getvalue())

    def test_on_epoch_begin_updates_variable_loss(self):
        """Variable losses are updated at the beginning of every epoch."""
        flow = build_flow(loss_mode='random')
        with patch.object(flow.loss, 'update_lambda_values_on_epoch_begin') as update:
            flow.on_epoch_begin(3, logs={})
        update.assert_called_once_with(3, logs=flow.log)
        flow = build_flow()
        flow.on_epoch_begin(0, logs={})

    def test_training_without_validation_samples(self):
        """Training without validation samples records NaN validation losses."""
        flow = build_flow(validation_split=0.0)
        self.assertEqual(flow.num_test_samples, 0)
        history = flow.train(**SHORT_TRAINING)
        self.assertEqual(len(history.history['loss']), SHORT_TRAINING['epochs'])
        self.assertTrue(np.all(np.isnan(flow.log['val_loss'])))

#########################################################################################################
# Training monitoring


class TestTrainingMonitoring(PlotTestCase):
    """Training metrics and epoch hooks."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_initial_chi2(self):
        """The pre-flow chi2 statistics are computed on the validation samples."""
        flow = build_flow()
        self.assertEqual(len(flow.chi2Y), flow.num_test_samples)
        self.assertTrue(0.0 <= flow.chi2Y_ks_p <= 1.0)
        self.assertEqual(set(flow.log.keys()), set(flow.training_metrics))
        self.assertTrue(all(len(_v) == 0 for _v in flow.log.values()))

    def test_initial_chi2_single_validation_sample(self):
        """With one validation sample no covariance is estimated and the chi2 is zero."""
        chain = make_gaussian_chain(num_samples=50)
        flow = build_flow(chain, validation_training_idx=(np.array([0]), np.arange(1, 50)))
        self.assertEqual(flow.num_test_samples, 1)
        np.testing.assert_allclose(flow.chi2Y, [0.0], atol=tolerance())

    def test_initial_chi2_non_finite_covariance(self):
        """A degenerate validation covariance falls back to unwhitened chi2 values."""
        flow = build_flow()
        weights = np.zeros(flow.num_test_samples)
        weights[0] = 1.0
        flow.test_weights = weights
        flow.has_weights = False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flow._init_training_monitoring()
        expected = np.sum((flow.test_samples - flow.test_samples[0])**2, axis=1)
        np.testing.assert_allclose(flow.chi2Y, expected, rtol=tolerance())
        # with weights, the chi2 values are resampled from the only sample with non-zero weight:
        flow.has_weights = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flow._init_training_monitoring()
        np.testing.assert_array_equal(flow.chi2Y, np.zeros(flow.num_test_samples))

    def test_initial_chi2_negative_weight_sum(self):
        """Weights with a negative sum give unwhitened chi2 values resampled uniformly."""
        flow = build_flow()
        weights = np.ones(flow.num_test_samples)
        weights[0] = -2.0 * flow.num_test_samples
        flow.test_weights = weights
        flow.has_weights = True
        flow._init_training_monitoring()
        mean = np.average(flow.test_samples, axis=0, weights=weights)
        values = np.sum((flow.test_samples - mean)**2, axis=1)
        self.assertEqual(len(flow.chi2Y), flow.num_test_samples)
        distances = np.abs(flow.chi2Y[:, None] - values[None, :])
        self.assertTrue(np.all(np.amin(distances, axis=1) <= tolerance() * np.amax(values)))
        self.assertTrue(0.0 <= flow.chi2Y_ks_p <= 1.0)

    def test_ks_test_negative_weight_sum(self):
        """In the training metrics, weights with a negative sum resample chi2Z uniformly."""
        flow = build_flow()
        weights = np.ones(flow.num_test_samples)
        weights[0] = -2.0 * flow.num_test_samples
        flow.test_weights = weights
        flow.has_weights = True
        flow.compute_training_metrics(logs={'loss': 1.0, 'val_loss': 0.8, 'lr': 1.e-3})
        values = np.sum(flow._trainable_inverse(flow.test_samples)**2, axis=1)
        self.assertEqual(flow.chi2Z.shape, (flow.num_test_samples,))
        self.assertTrue(np.all(np.isin(flow.chi2Z, values)))
        self.assertTrue(0.0 <= flow.log['chi2Z_ks_p'][-1] <= 1.0)

    def test_ks_test_failure_is_logged_as_zero(self):
        """A failing KS test logs zero statistics instead of stopping the training."""
        flow = build_flow()
        with patch.object(sp.scipy.stats, 'kstest', side_effect=ValueError('failed')):
            flow.compute_training_metrics(logs={'loss': 1.0, 'val_loss': 0.8, 'lr': 1.e-3})
        self.assertEqual(flow.log['chi2Z_ks'], [0.0])
        self.assertEqual(flow.log['chi2Z_ks_p'], [0.0])
        self.assertEqual(flow.log['loss'], [1.0])

    def test_compute_training_metrics(self):
        """Metrics are appended to the log, with loss rates as differences."""
        flow = build_flow()
        flow.compute_training_metrics(logs={'loss': 1.0, 'val_loss': 0.8, 'lr': 1.e-3})
        flow.compute_training_metrics(logs={'loss': 0.5, 'val_loss': 0.6, 'lr': 1.e-3})
        self.assertEqual(flow.log['loss'], [1.0, 0.5])
        self.assertEqual(flow.log['val_loss'], [0.8, 0.6])
        self.assertEqual(flow.log['lr'], [1.e-3, 1.e-3])
        self.assertEqual(flow.log['loss_rate'], [0.0, -0.5])
        self.assertEqual(flow.log['val_loss_rate'][0], 0.0)
        self.assertAlmostEqual(flow.log['val_loss_rate'][1], -0.2)
        self.assertEqual(len(flow.log['chi2Z_ks']), 2)
        self.assertEqual(flow.chi2Z.shape, (flow.num_test_samples,))

    def test_compute_training_metrics_posterior_losses(self):
        """Posterior losses log the loss breakdown, evidences and lambdas."""
        flow = build_flow(loss_mode='fixed')
        flow.compute_training_metrics(logs={'loss': 1.0, 'val_loss': 0.8, 'lr': 1.e-3})
        for key in ['rho_loss', 'ee_loss', 'val_rho_loss', 'val_ee_loss', 'evidence', 'evidence_error',
                    'training_evidence', 'test_evidence', 'rho_loss_rate', 'ee_loss_rate']:
            self.assertEqual(len(flow.log[key]), 1, key)
        evidence, error = flow.evidence()
        self.assertAlmostEqual(flow.log['evidence'][0], evidence, places=5)
        self.assertAlmostEqual(flow.log['evidence_error'][0], error, places=5)
        training_evidence, _ = flow.evidence(indexes=flow.training_idx)
        self.assertAlmostEqual(flow.log['training_evidence'][0], training_evidence, places=5)
        flow = build_flow(loss_mode='random')
        flow.compute_training_metrics(logs={'loss': 1.0, 'val_loss': 0.8, 'lr': 1.e-3})
        flow.compute_training_metrics(logs={'loss': 0.9, 'val_loss': 0.7, 'lr': 1.e-3})
        self.assertEqual(len(flow.log['lambda_1']), 2)
        self.assertEqual(len(flow.log['rho_loss_rate']), 2)

    def test_on_epoch_end_computes_metrics(self):
        """on_epoch_end fills the logs and a missing learning rate becomes 0."""
        flow = build_flow()
        logs = {'loss': 0.3, 'val_loss': 0.2}
        flow.on_epoch_end(0, logs=logs)
        self.assertEqual(flow.log['loss'], [0.3])
        self.assertEqual(flow.log['lr'], [0.0])
        flow = build_flow()
        flow.on_epoch_end(0, logs=None)
        self.assertEqual(flow.log['lr'], [0.0])
        self.assertEqual(flow.log['loss_rate'], [0.0])

    def test_on_epoch_end_high_feedback_copies_metrics(self):
        """At feedback 3 the metrics are copied into the epoch logs."""
        flow = build_flow()
        flow.feedback = 3
        logs = {'loss': 0.3, 'val_loss': 0.2, 'lr': 1.e-3}
        flow.on_epoch_end(0, logs=logs)
        self.assertIn('chi2Z_ks', logs)
        self.assertEqual(logs['loss_rate'], 0.0)

    def test_on_epoch_end_plotting(self):
        """Training plots are made every plot_every epochs, unless on a cluster backend."""
        flow = build_flow()
        flow.feedback = 1
        flow.plot_every = 2
        logs = {'loss': 0.3, 'val_loss': 0.2, 'lr': 1.e-3}
        with patch.object(sp, 'cluster_plotting', False), patch.object(sp, 'ipython_plotting', False), \
                patch.object(flow, 'training_plot') as training_plot:
            flow.on_epoch_end(0, logs=dict(logs))
            self.assertEqual(training_plot.call_count, 0)
            flow.on_epoch_end(1, logs=dict(logs))
            self.assertEqual(training_plot.call_count, 1)
        with patch.object(sp, 'cluster_plotting', True), patch.object(flow, 'training_plot') as training_plot:
            flow.on_epoch_end(3, logs=dict(logs))
            self.assertEqual(training_plot.call_count, 0)
        with patch.object(sp, 'cluster_plotting', False), patch.object(flow, 'training_plot') as training_plot:
            flow.feedback = 0
            flow.on_epoch_end(5, logs=dict(logs))
            self.assertEqual(training_plot.call_count, 0)

    def test_on_epoch_end_plotting_ipython(self):
        """In ipython mode the output is cleared and a new figure is made."""
        flow = build_flow()
        flow.feedback = 1
        flow.plot_every = 1
        logs = {'loss': 0.3, 'val_loss': 0.2, 'lr': 1.e-3}
        with patch.object(sp, 'cluster_plotting', False), patch.object(sp, 'ipython_plotting', True), \
                patch.object(sp, 'clear_output', create=True) as clear_output, \
                patch.object(flow, 'training_plot') as training_plot:
            flow.on_epoch_end(0, logs=logs)
        clear_output.assert_called_once_with(wait=True)
        training_plot.assert_called_once()
        self.assertTrue(hasattr(flow, 'fig'))

    def test_on_epoch_end_interactive_backend(self):
        """With an interactive backend the figure is redrawn and shown after plotting."""
        flow = build_flow()
        flow.feedback = 1
        flow.plot_every = 1
        logs = {'loss': 0.3, 'val_loss': 0.2, 'lr': 1.e-3}
        with patch.object(sp, 'cluster_plotting', False), patch.object(sp, 'ipython_plotting', False), \
                patch.object(sp.matplotlib, 'get_backend', return_value='MacOSX'), \
                patch.object(sp.plt, 'clf') as clf, patch.object(sp.plt, 'pause') as pause, \
                patch.object(sp.plt, 'show') as show, patch.object(flow, 'training_plot') as training_plot:
            flow.on_epoch_end(0, logs=logs)
        clf.assert_called_once_with()
        training_plot.assert_called_once()
        pause.assert_called_once()
        show.assert_called_once_with()

    def test_on_train_begin_and_end(self):
        """Outside ipython the figure is created and deleted around training."""
        flow = build_flow()
        with patch.object(sp, 'ipython_plotting', False):
            flow.on_train_begin(logs={})
            self.assertTrue(hasattr(flow, 'fig'))
            flow.on_train_end(logs={})
            self.assertFalse(hasattr(flow, 'fig'))
        with patch.object(sp, 'ipython_plotting', True):
            flow.on_train_begin(logs={})
            self.assertFalse(hasattr(flow, 'fig'))
            flow.on_train_end(logs={})

#########################################################################################################
# Plotting helpers


class TestPlotting(PlotTestCase):
    """Plot helpers and training_plot on trained flows."""

    @classmethod
    def setUpClass(cls):
        """Train one small flow per loss family."""
        seed_everything(0)
        cls.flows = {}
        for mode in ['standard', 'fixed', 'random']:
            flow = build_flow(loss_mode=mode)
            flow.train(epochs=3, steps_per_epoch=2, batch_size=32)
            cls.flows[mode] = flow

    def tearDown(self):
        """Close all figures and drop the flow figures."""
        for flow in self.flows.values():
            flow.__dict__.pop('fig', None)
        super().tearDown()

    def test_create_figure(self):
        """A figure is created for every loss family."""
        for flow in self.flows.values():
            flow._create_figure()
            self.assertIsInstance(flow.fig, matplotlib.figure.Figure)

    def test_basic_plot_helpers(self):
        """Loss, learning rate, chi2 and KS plots draw on the axes."""
        flow = self.flows['standard']
        fig, axes = plt.subplots(2, 3)
        flow._plot_loss(axes[0, 0])
        flow._plot_lr(axes[0, 1])
        flow._plot_chi2_dist(axes[0, 2], fast=True)
        flow._plot_chi2_dist(axes[1, 0], fast=False)
        flow._plot_chi2_ks_p(axes[1, 1])
        flow._plot_losses_rate(axes[1, 2])
        self.assertEqual(len(axes[0, 0].get_lines()), 2)
        self.assertEqual(axes[0, 0].get_yscale(), 'log')
        self.assertEqual(axes[0, 1].get_yscale(), 'log')
        self.assertGreater(len(axes[1, 0].patches), len(axes[0, 2].patches))
        plt.close(fig)

    def test_plot_loss_negative_and_best_loss(self):
        """Negative losses use a symlog scale and the population best is drawn."""
        flow = build_flow()
        flow.log['loss'] = [1.0, -0.001]
        flow.log['val_loss'] = [1.2, -0.002]
        flow.log['best_loss'] = 0.5
        fig, ax = plt.subplots()
        flow._plot_loss(ax)
        self.assertEqual(ax.get_yscale(), 'symlog')
        self.assertEqual(len(ax.get_lines()), 4)
        plt.close(fig)

    def test_plot_losses_rate_all_losses(self):
        """Loss rate plots work with and without absolute values for all losses."""
        for mode, flow in self.flows.items():
            fig, axes = plt.subplots(1, 2)
            flow._plot_losses_rate(axes[0], abs_value=False)
            flow._plot_losses_rate(axes[1], abs_value=True)
            self.assertEqual(axes[0].get_yscale(), 'symlog', mode)
            self.assertEqual(axes[1].get_yscale(), 'log', mode)
            plt.close(fig)

    def test_plot_losses_rate_long_log(self):
        """With a long log the symlog threshold uses the loss rate variance."""
        flow = build_flow()
        flow.log['loss_rate'] = list(np.linspace(-1.0, 1.0, 30))
        flow.log['val_loss_rate'] = list(np.linspace(-0.5, 0.5, 30))
        fig, ax = plt.subplots()
        flow._plot_losses_rate(ax, epoch_range=10)
        self.assertEqual(ax.get_yscale(), 'symlog')
        plt.close(fig)

    def test_plot_losses_rate_long_log_without_validation_rate(self):
        """Without a validation loss rate the symlog threshold uses the training loss rate."""
        flow = build_flow(loss_mode='fixed')
        self.assertNotIn('val_loss_rate', flow.log)
        rates = list(np.linspace(-1.0, 1.0, 30))
        for key in ['loss_rate', 'rho_loss_rate', 'ee_loss_rate']:
            flow.log[key] = rates
        fig, ax = plt.subplots()
        flow._plot_losses_rate(ax, epoch_range=10)
        self.assertEqual(ax.get_yscale(), 'symlog')
        self.assertAlmostEqual(ax.yaxis.get_transform().linthresh, 0.5 * np.sqrt(np.var(rates[-10:])))
        plt.close(fig)

    def test_posterior_loss_plot_helpers(self):
        """Loss breakdown, evidence and lambda plots draw on the axes."""
        fixed = self.flows['fixed']
        variable = self.flows['random']
        fig, axes = plt.subplots(2, 3)
        fixed._plot_density_evidence_error_losses(axes[0, 0])
        fixed._plot_evidence(axes[0, 1])
        fixed._plot_evidence_error(axes[0, 2])
        variable._plot_lambda_values(axes[1, 0])
        variable._plot_weighted_density_evidence_error_losses(axes[1, 1])
        variable._plot_density_evidence_error_losses(axes[1, 2])
        self.assertEqual(len(axes[0, 0].get_lines()), 4)
        self.assertEqual(len(axes[0, 1].get_lines()), 3)
        self.assertEqual(len(axes[1, 0].get_lines()), 2)
        self.assertEqual(len(axes[1, 1].get_lines()), 4)
        plt.close(fig)

    def test_training_plot_all_losses(self):
        """training_plot draws the panels of each loss family."""
        expected_axes = {'standard': 5, 'fixed': 8, 'random': 10}
        for mode, flow in self.flows.items():
            flow.training_plot(fast=True)
            self.assertEqual(len(flow.fig.axes), expected_axes[mode] + 1, mode)
            plt.close('all')

    def test_training_plot_title_and_population(self):
        """training_plot uses the title, or the population number."""
        flow = self.flows['standard']
        flow.training_plot(title='demo', fast=True)
        self.assertEqual(flow.fig._suptitle.get_text(), 'demo')
        plt.close('all')
        flow.log['population'] = 2
        try:
            flow.training_plot(fast=True)
            self.assertEqual(flow.fig._suptitle.get_text(), 'Training population 2')
        finally:
            flow.log.pop('population')

    def test_training_plot_replaces_and_reuses_figure(self):
        """An existing figure is replaced outside ipython and reused in ipython mode."""
        flow = self.flows['standard']
        flow._create_figure()
        first = flow.fig
        flow.training_plot(ipython_plotting=True, fast=True)
        self.assertIs(flow.fig, first)
        flow.training_plot(ipython_plotting=False, fast=True)
        self.assertIsNot(flow.fig, first)

    def test_training_plot_saves_file(self):
        """training_plot saves the figure to file_path."""
        flow = self.flows['random']
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'training.png')
            flow.training_plot(logs=flow.log, file_path=path, fast=True)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

#########################################################################################################
# Summary, sampling, evidence and smoothness


class TestSummaryAndSampling(unittest.TestCase):
    """print_training_summary, sample, MCSamples, evidence and smoothness_score."""

    @classmethod
    def setUpClass(cls):
        """Train one small flow."""
        seed_everything(0)
        cls.flow = build_trained_flow()

    def setUp(self):
        """Seed the random generators."""
        seed_everything(1)

    def test_print_training_summary(self):
        """The summary prints one line per metric with the last value."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.flow.print_training_summary()
        lines = output.getvalue().strip().split('\n')
        self.assertEqual(len(lines), len(self.flow.training_metrics))
        for line, metric in zip(lines, self.flow.training_metrics):
            self.assertTrue(line.startswith(metric))

    def test_print_training_summary_formats(self):
        """Small and large values use the exponential format."""
        flow = build_flow(trainable_bijector=None)
        flow.training_metrics = ['loss', 'val_loss', 'lr']
        flow.log = {'loss': [1.e-4], 'val_loss': [1.e4], 'lr': [0.5]}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            flow.print_training_summary()
        text = output.getvalue()
        self.assertIn('1.0000e-04', text)
        self.assertIn('1.0000e+04', text)
        self.assertIn('0.5000', text)

    def test_sample(self):
        """Samples are detached CPU tensors with the flow precision."""
        samples = self.flow.sample(50)
        self.assertEqual(tuple(samples.shape), (50, 2))
        self.assertEqual(samples.dtype, tu.get_precision())
        self.assertEqual(samples.device.type, 'cpu')
        self.assertFalse(samples.requires_grad)
        self.assertTrue(bool(torch.isfinite(samples).all()))

    def test_sample_is_seeded(self):
        """torch.manual_seed makes sampling reproducible."""
        torch.manual_seed(5)
        first = self.flow.sample(10)
        torch.manual_seed(5)
        second = self.flow.sample(10)
        self.assertTrue(torch.equal(first, second))

    def test_mcsamples_with_loglikes(self):
        """MCSamples carries names, labels, ranges and ``-log P``."""
        samples = self.flow.MCSamples(200)
        self.assertEqual(samples.samples.shape, (200, 2))
        self.assertEqual(samples.getParamNames().list(), ['p1', 'p2'])
        self.assertEqual([_p.label for _p in samples.getParamNames().names], ['p_1', 'p_2'])
        self.assertEqual(samples.name_tag, self.flow.name_tag)
        self.assertEqual(samples.ranges.getLower('p1'), self.flow.parameter_ranges['p1'][0])
        expected = -to_array(self.flow.log_probability(samples.samples))
        np.testing.assert_allclose(samples.loglikes, expected, rtol=10 * tolerance(), atol=10 * tolerance())

    def test_mcsamples_without_loglikes(self):
        """``logLikes=False`` gives samples without loglikes."""
        samples = self.flow.MCSamples(100, logLikes=False, name_tag='custom')
        self.assertEqual(samples.samples.shape, (100, 2))
        self.assertIsNone(samples.loglikes)
        self.assertEqual(samples.name_tag, 'custom')

    def test_mcsamples_filters_non_finite(self):
        """Non-finite samples are removed."""
        flow = build_flow(trainable_bijector=None)
        flow.feedback = 1
        bad_samples = torch.tensor([[0.0, 0.0], [np.inf, 0.0], [0.1, 0.1]], dtype=tu.get_precision())
        with patch.object(flow, '_sample', return_value=bad_samples), quiet():
            samples = flow.MCSamples(3)
            samples_no_loglikes = flow.MCSamples(3, logLikes=False)
        self.assertEqual(samples.samples.shape, (2, 2))
        self.assertEqual(samples.loglikes.shape, (2,))
        self.assertEqual(samples_no_loglikes.samples.shape, (2, 2))

    def test_evidence(self):
        """The evidence estimate is finite and uses the requested samples."""
        average, error = self.flow.evidence()
        self.assertTrue(np.isfinite(average))
        self.assertTrue(np.isfinite(error))
        self.assertGreaterEqual(error, 0.0)
        indexes = np.arange(10)
        average_idx, _ = self.flow.evidence(indexes=indexes)
        flow_log_likes = to_array(self.flow.log_probability(self.flow.chain_samples[indexes]))
        expected = np.mean(-self.flow.chain_loglikes[indexes] - flow_log_likes)
        self.assertAlmostEqual(float(average_idx), expected, places=4)
        average_weighted, error_weighted = self.flow.evidence(weighted=True)
        self.assertTrue(np.isfinite(average_weighted))
        self.assertTrue(np.isfinite(error_weighted))
        average_both, _ = self.flow.evidence(indexes=indexes, weighted=True)
        self.assertTrue(np.isfinite(average_both))

    def test_evidence_of_exact_flow(self):
        """For the exact Gaussian the evidence scatter vanishes."""
        chain = make_gaussian_chain()
        flow = build_flow(chain, prior_bijector=None, trainable_bijector=None, validation_split=0.0)
        # replace the Gaussian approximation with the exact distribution:
        exact = bj.AffineTriL(GAUSSIAN_MEAN, np.linalg.cholesky(GAUSSIAN_COV))
        flow.bijector = bj.Chain([bj.Identity(), exact, bj.Identity()]).to(flow.device)
        flow._build_distributions()
        average, error = flow.evidence()
        expected = np.log(2.0 * np.pi) + 0.5 * np.log(np.linalg.det(GAUSSIAN_COV))
        self.assertAlmostEqual(float(average), expected, places=3)
        self.assertLess(float(error), 1.e-3)

    def test_evidence_needs_loglikes_of_flow_parameters(self):
        """Without chain log-likelihoods of the flow parameters the evidence raises a clear error."""
        flow_no_loglikes = build_flow(make_gaussian_chain(with_loglikes=False), prior_bijector=None,
                                      trainable_bijector=None)
        with self.assertRaises(ValueError):
            flow_no_loglikes.evidence()
        flow = build_flow(prior_bijector=None, trainable_bijector=None)
        analytical = sp.AnalyticalDerivedParamsBijector(['p1'], ['exp_p1'], ['e^{p_1}'],
                                                        forward_fn=torch.exp, inverse_fn=torch.log)
        with contextlib.redirect_stdout(io.StringIO()):
            transformed_flows = [sp.TransformedFlowCallback(flow, [bj.Identity(), bj.Scale(2.)]),
                                 sp.TransformedFlowCallback(flow, analytical)]
        for transformed in transformed_flows:
            with self.assertRaises(ValueError):
                transformed.evidence()
        self.assertTrue(np.isfinite(flow.evidence()[0]))

    def test_smoothness_score(self):
        """The smoothness score is finite and builds the nearest neighbours when needed."""
        flow = build_trained_flow()
        self.assertNotIn('chain_nearest_index', flow.__dict__)
        score = flow.smoothness_score()
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertIn('chain_nearest_index', flow.__dict__)
        flow = build_trained_flow(init_nearest=True)
        self.assertTrue(np.isfinite(flow.smoothness_score()))

#########################################################################################################
# Geometry API on a real flow


class TestGeometry(unittest.TestCase):
    """Information geometry methods on a trained flow."""

    @classmethod
    def setUpClass(cls):
        """Train a small flow and choose central evaluation points."""
        seed_everything(0)
        cls.flow = build_trained_flow()
        order = np.argsort(cls.flow.chain_loglikes)
        cls.points = cls.flow.chain_samples[order[:6]]
        cls.other_points = cls.flow.chain_samples[order[6:12]]

    def test_output_shapes(self):
        """Every geometry method has the documented shape."""
        n, d = self.points.shape
        expected = {
            'map_to_abstract_coord': (n, d),
            'log_det_metric': (n,),
            'direct_jacobian': (n, d, d),
            'inverse_jacobian': (n, d, d),
            'inverse_jacobian_coord_derivative': (n, d, d, d),
            'metric': (n, d, d),
            'inverse_metric': (n, d, d),
            'coord_metric_derivative': (n, d, d, d),
            'coord_inverse_metric_derivative': (n, d, d, d),
            'coord_metric_derivative_2': (n, d, d, d, d),
            'coord_inverse_metric_derivative_2': (n, d, d, d, d),
            'levi_civita_connection': (n, d, d, d),
            'log_probability': (n,),
            'log_probability_jacobian': (n, d),
            'log_probability_hessian': (n, d, d),
        }
        for name, shape in expected.items():
            result = getattr(self.flow, name)(self.points)
            self.assertEqual(tuple(result.shape), shape, name)
            self.assertEqual(result.dtype, tu.get_precision(), name)
            self.assertTrue(bool(torch.isfinite(result).all()), name)

    def test_map_round_trip(self):
        """Mapping to abstract coordinates and back is the identity."""
        abstract = self.flow.map_to_abstract_coord(self.points)
        back = self.flow.map_to_original_coord(abstract)
        np.testing.assert_allclose(to_array(back), self.points, atol=100 * tolerance())

    def test_metric_and_inverse_metric_are_inverses(self):
        """``metric @ inverse_metric`` is the identity."""
        metric = to_array(self.flow.metric(self.points))
        inverse_metric = to_array(self.flow.inverse_metric(self.points))
        identity = np.broadcast_to(np.eye(2), metric.shape)
        np.testing.assert_allclose(metric @ inverse_metric, identity, atol=100 * tolerance())
        np.testing.assert_allclose(metric, np.swapaxes(metric, -1, -2), rtol=10 * tolerance())

    def test_jacobians_are_inverses(self):
        """``direct_jacobian @ inverse_jacobian`` is the identity."""
        direct = to_array(self.flow.direct_jacobian(self.points))
        inverse = to_array(self.flow.inverse_jacobian(self.points))
        identity = np.broadcast_to(np.eye(2), direct.shape)
        np.testing.assert_allclose(direct @ inverse, identity, atol=100 * tolerance())
        metric = to_array(self.flow.metric(self.points))
        np.testing.assert_allclose(np.swapaxes(inverse, -1, -2) @ inverse, metric, rtol=100 * tolerance())

    def test_log_det_metric(self):
        """``log_det_metric`` is the log determinant of the metric."""
        log_det = to_array(self.flow.log_det_metric(self.points))
        metric = to_array(self.flow.metric(self.points))
        np.testing.assert_allclose(log_det, np.linalg.slogdet(metric)[1], atol=1000 * tolerance())

    def test_metric_derivative_is_symmetric(self):
        """The metric derivative is symmetric in the metric indices."""
        derivative = to_array(self.flow.coord_metric_derivative(self.points))
        np.testing.assert_allclose(derivative, np.swapaxes(derivative, 1, 2), atol=1000 * tolerance())

    def test_levi_civita_symmetric_lower_indices(self):
        """The Levi-Civita connection is symmetric in the lower indices."""
        connection = to_array(self.flow.levi_civita_connection(self.points))
        np.testing.assert_allclose(connection, np.swapaxes(connection, -1, -2), atol=1000 * tolerance())

    def test_geodesic_distance(self):
        """The geodesic distance is the Euclidean distance in abstract space."""
        abstract_1 = to_array(self.flow.map_to_abstract_coord(self.points))
        abstract_2 = to_array(self.flow.map_to_abstract_coord(self.other_points))
        total = self.flow.geodesic_distance(self.points, self.other_points)
        self.assertEqual(tuple(total.shape), ())
        self.assertAlmostEqual(float(total), np.linalg.norm(abstract_1 - abstract_2), places=3)
        per_point = self.flow.geodesic_distance(self.points, self.other_points, axis=-1)
        self.assertEqual(tuple(per_point.shape), (6,))
        np.testing.assert_allclose(to_array(per_point), np.linalg.norm(abstract_1 - abstract_2, axis=-1),
                                   atol=100 * tolerance())
        kept = self.flow.geodesic_distance(self.points, self.other_points, axis=-1, keepdims=True)
        self.assertEqual(tuple(kept.shape), (6, 1))

    def test_geodesic_bvp(self):
        """Geodesics start and end at the requested points."""
        trajectory = self.flow.geodesic_bvp(self.points, self.other_points, num_points=7)
        self.assertEqual(tuple(trajectory.shape), (6, 7, 2))
        np.testing.assert_allclose(to_array(trajectory[:, 0]), self.points, atol=100 * tolerance())
        np.testing.assert_allclose(to_array(trajectory[:, -1]), self.other_points, atol=100 * tolerance())
        abstract = to_array(self.flow.map_to_abstract_coord(trajectory.reshape(-1, 2))).reshape(6, 7, 2)
        steps = np.linalg.norm(np.diff(abstract, axis=1), axis=-1)
        np.testing.assert_allclose(steps, steps[:, :1] * np.ones_like(steps), atol=1000 * tolerance())

    def test_geodesic_ivp_not_implemented(self):
        """The initial value problem is not implemented."""
        with self.assertRaises(NotImplementedError):
            self.flow.geodesic_ivp(self.points, np.zeros_like(self.points), np.array([0.0, 1.0]))

    def test_log_probability_abs_consistency(self):
        """The abstract space log probability matches the parameter space one."""
        abstract = self.flow.map_to_abstract_coord(self.points)
        log_prob = to_array(self.flow.log_probability(self.points))
        log_prob_abs = to_array(self.flow.log_probability_abs(abstract))
        np.testing.assert_allclose(log_prob_abs, log_prob, atol=100 * tolerance())

    def test_log_probability_abs_derivatives(self):
        """The abstract space gradient follows the chain rule; the Hessian is symmetric."""
        abstract = self.flow.map_to_abstract_coord(self.points)
        gradient_abs = to_array(self.flow.log_probability_abs_jacobian(abstract))
        gradient = to_array(self.flow.log_probability_jacobian(self.points))
        direct = to_array(self.flow.direct_jacobian(self.points))
        expected = np.einsum('nij,ni->nj', direct, gradient)
        np.testing.assert_allclose(gradient_abs, expected, rtol=1000 * tolerance(), atol=1000 * tolerance())
        hessian_abs = to_array(self.flow.log_probability_abs_hessian(abstract))
        self.assertEqual(hessian_abs.shape, (6, 2, 2))
        np.testing.assert_allclose(hessian_abs, np.swapaxes(hessian_abs, -1, -2), atol=1000 * tolerance())

    def test_log_probability_gradient_matches_finite_differences(self):
        """The log probability gradient matches central finite differences."""
        if tu.get_precision() != torch.float64:
            step, atol = 1.e-2, 5.e-2
        else:
            step, atol = 1.e-5, 1.e-5
        gradient = to_array(self.flow.log_probability_jacobian(self.points))
        numerical = np.zeros_like(gradient)
        for i in range(2):
            shift = np.zeros(2)
            shift[i] = step
            plus = to_array(self.flow.log_probability(self.points + shift))
            minus = to_array(self.flow.log_probability(self.points - shift))
            numerical[:, i] = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(gradient, numerical, atol=atol, rtol=atol)

    def test_affine_flow_is_flat(self):
        """For an affine flow the metric is constant and the connection vanishes."""
        flow = build_flow(prior_bijector=None, trainable_bijector=None)
        derivative = to_array(flow.coord_metric_derivative(self.points))
        connection = to_array(flow.levi_civita_connection(self.points))
        np.testing.assert_allclose(derivative, np.zeros_like(derivative), atol=tolerance())
        np.testing.assert_allclose(connection, np.zeros_like(connection), atol=tolerance())
        second = to_array(flow.coord_inverse_metric_derivative_2(self.points))
        np.testing.assert_allclose(second, np.zeros_like(second), atol=tolerance())

    def test_direct_jacobian_of_input_without_gradients(self):
        """The device Jacobian accepts inputs whose abstract coordinates carry no graph."""
        flow = build_flow(trainable_bijector=None)
        x = torch.as_tensor(self.points, dtype=tu.get_precision())
        self.assertFalse(flow._map_to_abstract_coord(x).requires_grad)
        jacobian = to_array(flow._direct_jacobian(x, create_graph=False))
        np.testing.assert_allclose(jacobian, to_array(flow.direct_jacobian(self.points)), rtol=tolerance())
        product = np.einsum('...ij,...jk->...ik', jacobian, to_array(flow.inverse_jacobian(self.points)))
        np.testing.assert_allclose(product, np.broadcast_to(np.eye(2), product.shape), atol=100 * tolerance())

#########################################################################################################
# Public outputs: detached CPU tensors or graph-attached tensors


class TestPublicOutputs(unittest.TestCase):
    """Conversion rule of the public methods."""

    @classmethod
    def setUpClass(cls):
        """Train a small flow."""
        seed_everything(0)
        cls.flow = build_trained_flow()
        cls.points = cls.flow.chain_samples[:4]

    def assert_detached_cpu(self, result, name):
        """Check a detached CPU tensor with the active precision."""
        self.assertTrue(torch.is_tensor(result), name)
        self.assertFalse(result.requires_grad, name)
        self.assertEqual(result.device.type, 'cpu', name)
        self.assertEqual(result.dtype, tu.get_precision(), name)

    def test_numpy_list_and_tensor_inputs(self):
        """Numpy, list and tensor inputs give the same detached CPU results."""
        methods = ['log_probability', 'log_probability_jacobian', 'map_to_abstract_coord', 'metric',
                   'log_det_metric']
        for name in methods:
            method = getattr(self.flow, name)
            from_numpy = method(self.points)
            from_list = method(self.points.tolist())
            from_tensor = method(torch.as_tensor(self.points))
            for result in (from_numpy, from_list, from_tensor):
                self.assert_detached_cpu(result, name)
            np.testing.assert_allclose(to_array(from_numpy), to_array(from_list), rtol=tolerance())
            np.testing.assert_allclose(to_array(from_numpy), to_array(from_tensor), rtol=tolerance())
            self.assertIsInstance(from_numpy.numpy(), np.ndarray)

    def test_sample_and_geodesics_detached(self):
        """Samples and geodesics are detached CPU tensors."""
        self.assert_detached_cpu(self.flow.sample(3), 'sample')
        self.assert_detached_cpu(self.flow.geodesic_bvp(self.points[:2], self.points[2:], num_points=3), 'bvp')
        self.assert_detached_cpu(self.flow.geodesic_distance(self.points[:2], self.points[2:]), 'distance')

    def test_graph_attached_log_probability(self):
        """A tensor input requiring gradients gives a graph-attached result."""
        x = torch.as_tensor(self.points, dtype=tu.get_precision()).requires_grad_(True)
        log_prob = self.flow.log_probability(x)
        self.assertTrue(log_prob.requires_grad)
        self.assertEqual(log_prob.device, self.flow.device)
        gradient, = torch.autograd.grad(log_prob.sum(), x)
        expected = self.flow.log_probability_jacobian(self.points)
        np.testing.assert_allclose(to_array(gradient), to_array(expected), atol=10 * tolerance())

    def test_graph_attached_derivatives_nest(self):
        """Derivative methods stay attached and can be differentiated again."""
        x = torch.as_tensor(self.points, dtype=tu.get_precision()).requires_grad_(True)
        gradient = self.flow.log_probability_jacobian(x)
        self.assertTrue(gradient.requires_grad)
        self.assertEqual(gradient.device, self.flow.device)
        second = torch.stack(
            [torch.autograd.grad(gradient[:, i].sum(), x, retain_graph=True)[0] for i in range(2)], dim=1)
        hessian = self.flow.log_probability_hessian(self.points)
        np.testing.assert_allclose(to_array(second), to_array(hessian), atol=100 * tolerance())
        metric = self.flow.metric(x)
        self.assertTrue(metric.requires_grad)
        metric_gradient, = torch.autograd.grad(metric.sum(), x)
        self.assertEqual(tuple(metric_gradient.shape), (4, 2))

    def test_graph_attached_geodesics(self):
        """Geodesic methods keep the graph when an input requires gradients."""
        x = torch.as_tensor(self.points[:2], dtype=tu.get_precision()).requires_grad_(True)
        distance = self.flow.geodesic_distance(x, self.points[2:])
        self.assertTrue(distance.requires_grad)
        trajectory = self.flow.geodesic_bvp(x, self.points[2:], num_points=3)
        self.assertTrue(trajectory.requires_grad)

    def test_tensor_without_grad_is_detached(self):
        """A tensor input without gradients gives a detached result."""
        x = torch.as_tensor(self.points, dtype=tu.get_precision())
        for name in ['log_probability', 'log_probability_jacobian', 'direct_jacobian', 'log_probability_abs']:
            self.assert_detached_cpu(getattr(self.flow, name)(x), name)
        self.assertFalse(x.requires_grad)

#########################################################################################################
# Derived parameters bijector


class TestDerivedParamsBijector(unittest.TestCase):
    """Trainable map between two parameterizations."""

    @classmethod
    def setUpClass(cls):
        """Build and train a derived parameters bijector swapping the two parameters."""
        seed_everything(0)
        cls.chain = make_gaussian_chain()
        cls.derived = sp.DerivedParamsBijector(
            cls.chain, ['p1', 'p2'], ['p2', 'p1'], permutations=False, feedback=0,
            hidden_units=[8], n_transformations=1)
        cls.history = cls.derived.train(epochs=5, steps_per_epoch=4, batch_size=32)

    def test_length_mismatch_raises(self):
        """Input and output parameters must have the same length."""
        with self.assertRaises(ValueError):
            sp.DerivedParamsBijector(self.chain, ['p1', 'p2'], ['p1'], feedback=0)

    def test_structure(self):
        """The bijector chains the output flow, the trainable map and the inverse input flow."""
        derived = self.derived
        self.assertEqual(derived.num_params, 2)
        self.assertIsInstance(derived.bijector, bj.Chain)
        self.assertIs(derived.bijector.bijectors[1], derived.trainable_bijector)
        self.assertIsInstance(derived.bijector.bijectors[2], bj.Invert)
        self.assertIsInstance(derived.trainer.loss, lf.mean_squared_error)
        np.testing.assert_array_equal(derived.flow_in.test_idx, derived.flow_out.test_idx)
        self.assertEqual(derived.num_training_samples, derived.flow_in.num_training_samples)
        np.testing.assert_allclose(derived.chain_samples, self.chain.samples[:, [1, 0]].astype(tu.np_prec))

    def test_training_history(self):
        """Training returns finite losses per epoch."""
        for key in ['loss', 'val_loss', 'lr']:
            self.assertEqual(len(self.history.history[key]), 5)
        self.assertTrue(np.all(np.isfinite(self.history.history['loss'])))
        self.assertTrue(np.all(np.isfinite(self.history.history['val_loss'])))

    def test_training_reduces_validation_loss(self):
        """More training reduces the mean squared error."""
        seed_everything(2)
        derived = sp.DerivedParamsBijector(
            self.chain, ['p1', 'p2'], ['p2', 'p1'], feedback=0, hidden_units=[8], n_transformations=1)
        history = derived.train(epochs=30, steps_per_epoch=5, batch_size=64)
        self.assertLess(history.history['val_loss'][-1], history.history['val_loss'][0])

    def test_train_default_batches_and_verbosity(self):
        """Missing batch options are derived from each other and verbosity follows the feedback level."""
        derived = self.derived
        num_samples = derived.num_training_samples
        with patch.object(derived.trainer, 'fit') as fit:
            history = derived.train(epochs=1)
        self.assertIs(history, fit.return_value)
        self.assertEqual(fit.call_args.kwargs['steps_per_epoch'], 20)
        self.assertEqual(fit.call_args.kwargs['batch_size'], int(num_samples / 20))
        self.assertEqual(fit.call_args.kwargs['verbose'], 0)
        with patch.object(derived.trainer, 'fit') as fit:
            derived.train(epochs=1, batch_size=50)
        self.assertEqual(fit.call_args.kwargs['batch_size'], 50)
        self.assertEqual(fit.call_args.kwargs['steps_per_epoch'], int(num_samples / 50))
        with patch.object(derived, 'feedback', 3), patch.object(derived.trainer, 'fit') as fit:
            derived.train(epochs=1)
        self.assertEqual(fit.call_args.kwargs['verbose'], 1)

    def test_partial_transformation_ranges(self):
        """A derived map of a subset of parameters keeps the ranges of the other parameters."""
        seed_everything(0)
        chain = make_three_param_chain()
        flow = build_flow(chain, trainable_bijector=None)
        derived = sp.DerivedParamsBijector(chain, ['a', 'b'], ['b', 'a'], feedback=0, hidden_units=[8],
                                           n_transformations=1)
        transformed = sp.TransformedFlowCallback(flow, derived)
        self.assertEqual(transformed.param_names, ['b', 'a', 'c'])
        self.assertEqual(transformed.param_labels, ['b', 'a', 'c'])
        self.assertEqual(transformed.parameter_ranges['c'], flow.parameter_ranges['c'])
        self.assertEqual(transformed.parameter_ranges['a'], derived.flow_out.parameter_ranges['a'])
        np.testing.assert_array_equal(transformed.chain_samples[:, 2], flow.chain_samples[:, 2])
        output_ranges = derived.flow_out.parameter_ranges
        derived.flow_out.parameter_ranges = None
        self.assertIsNone(sp.TransformedFlowCallback(flow, derived).parameter_ranges)
        derived.flow_out.parameter_ranges = output_ranges
        flow.parameter_ranges = None
        self.assertIsNone(sp.TransformedFlowCallback(flow, derived).parameter_ranges)

    def test_transformed_flow(self):
        """A flow transformed with the derived bijector uses the output parameters."""
        seed_everything(0)
        flow = build_trained_flow(self.chain)
        transformed = sp.TransformedFlowCallback(flow, self.derived)
        self.assertEqual(transformed.param_names, ['p2', 'p1'])
        self.assertEqual(transformed.param_labels, ['p_2', 'p_1'])
        self.assertEqual(set(transformed.parameter_ranges.keys()), {'p1', 'p2'})
        self.assertEqual(transformed.chain_samples.shape, (500, 2))
        self.assertFalse(transformed.has_loglikes)
        log_prob = transformed.log_probability(self.chain.samples[:5, [1, 0]])
        self.assertTrue(bool(torch.isfinite(log_prob).all()))

#########################################################################################################
# Transformed flows


class TestTransformedFlowCallback(unittest.TestCase):
    """Flows transformed with elementwise, analytical and derived maps."""

    @classmethod
    def setUpClass(cls):
        """Train a small flow on a Gaussian chain."""
        seed_everything(0)
        cls.chain = make_gaussian_chain()
        cls.flow = build_trained_flow(cls.chain)
        cls.points = cls.chain.samples[:5]

    def test_elementwise_transformation(self):
        """A list of elementwise bijectors renames and rescales the parameters."""
        transformed = sp.TransformedFlowCallback(self.flow, [bj.Scale(2.0, name='double'), bj.Identity(name='')])
        self.assertEqual(transformed.param_names, ['double_p1', 'p2'])
        self.assertEqual(transformed.param_labels, ['double p_1', 'p_2'])
        self.assertEqual(transformed.name_tag, self.flow.name_tag + '_transformed')
        self.assertAlmostEqual(float(transformed.parameter_ranges['double_p1'][0]),
                               2.0 * self.flow.parameter_ranges['p1'][0], places=4)
        np.testing.assert_allclose(transformed.chain_samples, self.flow.chain_samples * np.array([2.0, 1.0]),
                                   rtol=tolerance())
        np.testing.assert_allclose(transformed.sample_MAP[0], self.flow.sample_MAP * np.array([2.0, 1.0]),
                                   rtol=tolerance())
        self.assertIsNone(transformed.chain_loglikes)
        self.assertFalse(transformed.has_loglikes)
        self.assertEqual(transformed.num_params, 2)
        self.assertIs(transformed.trainable_bijector, self.flow.trainable_bijector)

    def test_elementwise_log_probability(self):
        """The transformed density includes the Jacobian of the transformation."""
        transformed = sp.TransformedFlowCallback(self.flow, [bj.Scale(2.0), bj.Identity()])
        self.assertEqual(transformed.param_names, ['scale_p1', 'identity_p2'])
        log_prob = to_array(transformed.log_probability(self.points * np.array([2.0, 1.0])))
        expected = to_array(self.flow.log_probability(self.points)) - np.log(2.0)
        np.testing.assert_allclose(log_prob, expected, atol=100 * tolerance())

    def test_transform_posterior_false(self):
        """Without transforming the posterior the density keeps the original values."""
        transformed = sp.TransformedFlowCallback(self.flow, [bj.Scale(2.0), bj.Shift(1.0)],
                                                 transform_posterior=False)
        self.assertFalse(transformed.transform_posterior)
        log_prob = to_array(transformed.log_probability(self.points * np.array([2.0, 1.0]) + np.array([0.0, 1.0])))
        expected = to_array(self.flow.log_probability(self.points))
        np.testing.assert_allclose(log_prob, expected, atol=100 * tolerance())

    def test_sampling_and_geometry(self):
        """Transformed flows sample and support the geometry API."""
        transformed = sp.TransformedFlowCallback(self.flow, [bj.Scale(2.0), bj.Identity()])
        samples = transformed.sample(20)
        self.assertEqual(tuple(samples.shape), (20, 2))
        mc_samples = transformed.MCSamples(20)
        self.assertEqual(mc_samples.getParamNames().list(), transformed.param_names)
        metric = transformed.metric(self.points * np.array([2.0, 1.0]))
        self.assertEqual(tuple(metric.shape), (5, 2, 2))

    def test_map_coordinates_are_transformed(self):
        """MAP coordinates and values are carried over."""
        flow = build_trained_flow(self.chain)
        flow.MAP_coord = flow.sample_MAP.copy()
        flow.MAP_logP = float(flow.log_probability(np.atleast_2d(flow.MAP_coord))[0])
        transformed = sp.TransformedFlowCallback(flow, [bj.Shift(1.0), bj.Identity()])
        np.testing.assert_allclose(transformed.MAP_coord, flow.MAP_coord + np.array([1.0, 0.0]), rtol=tolerance())
        self.assertAlmostEqual(transformed.MAP_logP, flow.MAP_logP, places=4)

    def test_missing_ranges_and_maps(self):
        """Missing ranges and sample MAP stay missing, the chain MAP is transformed."""
        flow = build_flow(self.chain, trainable_bijector=None)
        flow.parameter_ranges = None
        flow.sample_MAP = None
        flow.chain_MAP = np.array([0.1, -0.2])
        transformed = sp.TransformedFlowCallback(flow, [bj.Shift(1.0), bj.Scale(2.0)])
        self.assertIsNone(transformed.parameter_ranges)
        self.assertIsNone(transformed.sample_MAP)
        np.testing.assert_allclose(transformed.chain_MAP, [[1.1, -0.4]], rtol=tolerance())

    def test_analytical_transformation_without_loglikes_and_ranges(self):
        """Analytical transformations of flows without loglikes and ranges."""
        flow = build_flow(make_gaussian_chain(with_loglikes=False), trainable_bijector=None)
        flow.parameter_ranges = None
        analytical = sp.AnalyticalDerivedParamsBijector(
            ['p1', 'p2'], ['s', 'd'], ['s', 'd'],
            forward_fn=sum_difference_forward, inverse_fn=sum_difference_inverse)
        transformed = sp.TransformedFlowCallback(flow, analytical)
        self.assertIsNone(transformed.chain_loglikes)
        self.assertFalse(transformed.has_loglikes)
        self.assertIsNone(transformed.parameter_ranges)
        samples = flow.chain_samples.astype(np.float64)
        np.testing.assert_allclose(transformed.chain_samples[:, 0], samples[:, 0] + samples[:, 1],
                                   atol=10 * tolerance())

    def test_wrong_number_of_bijectors_raises(self):
        """One bijector per parameter is needed."""
        with self.assertRaises(ValueError):
            sp.TransformedFlowCallback(self.flow, [bj.Scale(2.0)])

    def test_unsupported_transformation_raises(self):
        """Other transformation types raise ValueError."""
        with self.assertRaises(ValueError):
            sp.TransformedFlowCallback(self.flow, 2.0)

    def test_cannot_be_trained(self):
        """Transformed flows cannot be trained."""
        transformed = sp.TransformedFlowCallback(self.flow, [bj.Identity(), bj.Identity()])
        with self.assertRaises(NotImplementedError):
            transformed.train()
        with self.assertRaises(NotImplementedError):
            transformed.global_train()

    def test_analytical_length_mismatch_raises(self):
        """Input and output parameters must have the same length."""
        with self.assertRaises(ValueError):
            sp.AnalyticalDerivedParamsBijector(['p1', 'p2'], ['s'], ['s'],
                                               forward_fn=sum_difference_forward,
                                               inverse_fn=sum_difference_inverse)

    def test_analytical_bijector(self):
        """The analytical transformation wraps an Inline bijector."""
        analytical = sp.AnalyticalDerivedParamsBijector(
            ['p1', 'p2'], ['s', 'd'], ['s', 'd'],
            forward_fn=sum_difference_forward, inverse_fn=sum_difference_inverse, not_an_option=1)
        self.assertIsInstance(analytical.bijector, bj.Inline)
        self.assertEqual(analytical.bijector.min_event_ndims, 1)
        self.assertEqual(analytical.num_params, 2)
        x = torch.as_tensor(self.points, dtype=tu.get_precision())
        with torch.no_grad():
            back = analytical.bijector.inverse(analytical.bijector.forward(x))
            log_det = analytical.bijector.forward_log_det_jacobian(x, event_ndims=1)
        np.testing.assert_allclose(to_array(back), self.points, atol=10 * tolerance())
        np.testing.assert_allclose(to_array(log_det), np.log(2.0) * np.ones(5), atol=10 * tolerance())

    def test_analytical_transformation(self):
        """An analytical transformation of all parameters."""
        analytical = sp.AnalyticalDerivedParamsBijector(
            ['p1', 'p2'], ['s', 'd'], ['s', 'd'],
            forward_fn=sum_difference_forward, inverse_fn=sum_difference_inverse)
        transformed = sp.TransformedFlowCallback(self.flow, analytical)
        self.assertEqual(transformed.param_names, ['s', 'd'])
        self.assertEqual(transformed.param_labels, ['s', 'd'])
        samples = self.flow.chain_samples.astype(np.float64)
        expected_samples = np.stack([samples[:, 0] + samples[:, 1], samples[:, 0] - samples[:, 1]], axis=1)
        np.testing.assert_allclose(transformed.chain_samples, expected_samples, atol=10 * tolerance())
        self.assertAlmostEqual(float(transformed.parameter_ranges['s'][0]), np.amin(expected_samples[:, 0]), places=4)
        self.assertAlmostEqual(float(transformed.parameter_ranges['d'][1]), np.amax(expected_samples[:, 1]), places=4)
        points = sum_difference_forward(torch.as_tensor(self.points, dtype=tu.get_precision()))
        log_prob = to_array(transformed.log_probability(points))
        expected = to_array(self.flow.log_probability(self.points)) - np.log(2.0)
        np.testing.assert_allclose(log_prob, expected, atol=100 * tolerance())

    def test_analytical_partial_transformation(self):
        """An analytical transformation of a subset of parameters puts them first."""
        analytical = sp.AnalyticalDerivedParamsBijector(
            ['p2'], ['exp_p2'], ['e^{p_2}'], forward_fn=torch.exp, inverse_fn=torch.log)
        transformed = sp.TransformedFlowCallback(self.flow, analytical)
        self.assertEqual(transformed.param_names, ['exp_p2', 'p1'])
        self.assertEqual(transformed.param_labels, ['e^{p_2}', 'p_1'])
        self.assertEqual(transformed.parameter_ranges['p1'], self.flow.parameter_ranges['p1'])
        samples = self.flow.chain_samples.astype(np.float64)
        np.testing.assert_allclose(transformed.chain_samples[:, 0], np.exp(samples[:, 1]), rtol=10 * tolerance())
        np.testing.assert_allclose(transformed.chain_samples[:, 1], samples[:, 0], rtol=tolerance())
        points = np.stack([np.exp(self.points[:, 1]), self.points[:, 0]], axis=1)
        log_prob = to_array(transformed.log_probability(points))
        expected = to_array(self.flow.log_probability(self.points)) - self.points[:, 1]
        np.testing.assert_allclose(log_prob, expected, atol=100 * tolerance())

#########################################################################################################
# Average flow


class TestAverageFlow(PlotTestCase):
    """Mixture of flows sharing the training split."""

    @classmethod
    def setUpClass(cls):
        """Train two small flows on the same split."""
        seed_everything(0)
        cls.chain = make_gaussian_chain()
        first = build_trained_flow(cls.chain)
        cls.split = (first.test_idx, first.training_idx)
        second = build_trained_flow(cls.chain, validation_training_idx=cls.split)
        cls.flows = [first, second]
        cls.points = cls.chain.samples[:5]

    def make_average(self, flows=None):
        """Average flow of the shared members."""
        if flows is None:
            flows = self.flows
        return sp.average_flow(flows, validation_training_idx=self.split)

    def test_default_weights(self):
        """Default weights follow the validation loss."""
        average = self.make_average()
        self.assertEqual(average.num_flows, 2)
        self.assertEqual(average.weights.dtype, tu.get_precision())
        self.assertEqual(average.weights.device.type, 'cpu')
        val_losses = np.array([_f.log['val_loss'][-1] for _f in self.flows])
        expected = np.exp(np.amin(val_losses) - val_losses)
        expected = expected / np.sum(expected)
        np.testing.assert_allclose(to_array(average.weights), expected, rtol=10 * tolerance())
        self.assertAlmostEqual(float(average.weights.sum()), 1.0, places=5)

    def test_weight_modes(self):
        """Weights for the equal, loss, KS p-value and generic modes."""
        average = self.make_average()
        average._set_flow_weights(mode='equal')
        np.testing.assert_allclose(to_array(average.weights), [0.5, 0.5])
        average._set_flow_weights(mode='loss')
        losses = np.array([_f.log['loss'][-1] for _f in self.flows])
        expected = np.exp(np.amin(losses) - losses)
        np.testing.assert_allclose(to_array(average.weights), expected / np.sum(expected), rtol=10 * tolerance())
        average._set_flow_weights(mode='chi2Z_ks_p')
        p_values = np.array([_f.log['chi2Z_ks_p'][-1] for _f in self.flows])
        np.testing.assert_allclose(to_array(average.weights), p_values / np.sum(p_values), rtol=10 * tolerance())
        average._set_flow_weights(mode='chi2Z_ks')
        values = np.array([_f.log['chi2Z_ks'][-1] for _f in self.flows])
        np.testing.assert_allclose(to_array(average.weights), values / np.sum(values), rtol=10 * tolerance())

    def test_missing_log_key_raises_value_error(self):
        """A missing or empty log entry raises ValueError (it used to be a NameError)."""
        average = self.make_average()
        with self.assertRaises(ValueError):
            average._set_flow_weights(mode='not_a_metric')
        empty = copy.copy(self.flows[1])
        empty.log = dict(empty.log, val_loss=[])
        with self.assertRaises(ValueError):
            self.make_average([self.flows[0], empty])

    def test_parameter_mismatch_raises(self):
        """Flows with different parameters cannot be averaged."""
        other = copy.copy(self.flows[1])
        other.param_names = ['p2', 'p1']
        with self.assertRaises(ValueError):
            self.make_average([self.flows[0], other])

    def test_device_mismatch_raises(self):
        """Flows on different devices cannot be averaged."""
        other = copy.copy(self.flows[1])
        other.device = torch.device('meta')
        with self.assertRaises(ValueError) as context:
            self.make_average([self.flows[0], other])
        self.assertIn('flow.to(device)', str(context.exception))

    def test_copied_attributes(self):
        """The average flow copies the chain information of the first member."""
        average = self.make_average()
        self.assertEqual(average.param_names, self.flows[0].param_names)
        self.assertEqual(average.name_tag, self.flows[0].name_tag)
        self.assertIs(average.chain_samples, self.flows[0].chain_samples)
        self.assertEqual(average.device, self.flows[0].device)
        np.testing.assert_array_equal(average.test_idx, self.split[0])
        np.testing.assert_array_equal(average.training_idx, self.split[1])

    def test_split_warnings(self):
        """Missing or inconsistent splits print a warning."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            average = sp.average_flow(self.flows)
        self.assertIn('validation_training_idx not found', output.getvalue())
        self.assertIsNone(average.test_idx)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sp.average_flow(self.flows, validation_training_idx=(self.split[0][::-1], self.split[1]))
        self.assertIn('not consistent', output.getvalue())

    def test_split_warning_with_different_sizes(self):
        """Splits of different sizes give the 'not consistent' warning."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sp.average_flow(self.flows, validation_training_idx=(self.split[1], self.split[0]))
        self.assertIn('not consistent', output.getvalue())

    def test_log_probability_is_mixture(self):
        """The log probability is the log-sum-exp of the weighted members."""
        average = self.make_average()
        log_prob = to_array(average.log_probability(self.points))
        log_weights = np.log(to_array(average.weights))
        members = np.stack([to_array(_f.log_probability(self.points)) for _f in self.flows], axis=1)
        expected = np.log(np.sum(np.exp(members + log_weights), axis=1))
        np.testing.assert_allclose(log_prob, expected, atol=10 * tolerance())
        gradient = average.log_probability_jacobian(self.points)
        self.assertEqual(tuple(gradient.shape), (5, 2))

    def test_average_of_transformed_flows(self):
        """Transformed members lack the fixed bijectors: this is reported and the mixture still works."""
        members = [sp.TransformedFlowCallback(_f, [bj.Shift(1.0), bj.Identity()]) for _f in self.flows]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            average = sp.average_flow(members, validation_training_idx=self.split)
        self.assertIn('Flow does not have attribute : prior_bijector', output.getvalue())
        self.assertNotIn('prior_bijector', average.__dict__)
        self.assertEqual(average.param_names, members[0].param_names)
        points = self.points + np.array([1.0, 0.0])
        log_weights = np.log(to_array(average.weights))
        values = np.stack([to_array(_f.log_probability(points)) for _f in members], axis=1)
        expected = np.log(np.sum(np.exp(values + log_weights), axis=1))
        np.testing.assert_allclose(to_array(average.log_probability(points)), expected, atol=10 * tolerance())

    def test_sampling(self):
        """The mixture samples on the host with the flow precision."""
        average = self.make_average()
        samples = average.sample(40)
        self.assertEqual(tuple(samples.shape), (40, 2))
        self.assertEqual(samples.device.type, 'cpu')
        self.assertEqual(samples.dtype, tu.get_precision())
        mc_samples = average.MCSamples(40)
        self.assertEqual(mc_samples.samples.shape, (40, 2))
        average._set_flow_weights(mode='equal')
        average.weights = tu.to_tensor([1.0, 0.0], device='cpu')
        average._build_distributions()
        torch.manual_seed(3)
        only_first = average.sample(10)
        torch.manual_seed(3)
        expected = self.flows[0].sample(10)
        self.assertEqual(tuple(only_first.shape), tuple(expected.shape))

    def test_abstract_space_not_available(self):
        """Average flows have no abstract coordinates."""
        average = self.make_average()
        for name in ['log_probability_abs', 'log_probability_abs_jacobian', 'log_probability_abs_hessian']:
            with self.assertRaises(NotImplementedError):
                getattr(average, name)(self.points)
        with self.assertRaises(NotImplementedError):
            average._log_probability_abs(self.points)

    def test_cast(self):
        """cast uses the first member."""
        average = self.make_average()
        value = average.cast([1.0])
        self.assertEqual(value.dtype, tu.get_precision())
        self.assertEqual(value.device.type, 'cpu')

    def test_print_training_summary(self):
        """The summary prints the weights and one line per metric."""
        average = self.make_average()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            average.print_training_summary()
        text = output.getvalue()
        self.assertIn('Number of flows: 2', text)
        self.assertIn('Flow weights', text)
        for metric in self.flows[0].training_metrics:
            self.assertIn(metric, text)

    def test_training_plot(self):
        """training_plot draws, and saves one file per member."""
        average = self.make_average()
        try:
            average.training_plot()
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, 'average.png')
                average.training_plot(file_path=path)
                self.assertTrue(os.path.isfile(os.path.join(folder, 'average_0.png')))
                self.assertTrue(os.path.isfile(os.path.join(folder, 'average_1.png')))
            with patch.object(sp.plt, 'show') as show:
                average.training_plot(ipython_plotting=True)
            self.assertEqual(show.call_count, 2)
        finally:
            for flow in self.flows:
                flow.__dict__.pop('fig', None)

    def test_train_and_global_train_forward_to_members(self):
        """Training an average flow trains every member."""
        average = self.make_average()
        with patch.object(sp.FlowCallback, 'train', autospec=True) as train:
            average.train(epochs=1)
        self.assertEqual([_c.args[0] for _c in train.call_args_list], self.flows)
        self.assertEqual(train.call_args.kwargs, {'epochs': 1})
        with patch.object(sp.FlowCallback, 'global_train', autospec=True) as global_train:
            average.global_train(pop_size=1)
        self.assertEqual(global_train.call_count, 2)

    def test_to_device(self):
        """to moves the members and returns the average flow."""
        average = self.make_average()
        self.assertIs(average.to(average.device), average)
        self.assertEqual(average.flows[0].device, average.device)
        log_prob = average.log_probability(self.points)
        self.assertEqual(tuple(log_prob.shape), (5,))

#########################################################################################################
# Cache helpers


class TestFlowFromChain(unittest.TestCase):
    """flow_from_chain and average_flow_from_chain."""

    def setUp(self):
        """Seed the random generators and create a temporary folder."""
        seed_everything(0)
        self.chain = make_gaussian_chain(num_samples=300)
        self.kwargs = dict(SMALL_FLOW, pop_size=1, **SHORT_TRAINING)
        self._folder = tempfile.TemporaryDirectory()
        self.folder = self._folder.name

    def tearDown(self):
        """Remove the temporary folder."""
        self._folder.cleanup()

    def test_flow_from_chain_without_cache(self):
        """Without cache the flow is built and trained with global_train."""
        with patch.object(sp.FlowCallback, 'global_train', autospec=True) as global_train:
            flow = sp.flow_from_chain(self.chain, **self.kwargs)
        self.assertIsInstance(flow, sp.FlowCallback)
        global_train.assert_called_once()
        self.assertEqual(global_train.call_args.kwargs['pop_size'], 1)
        self.assertEqual(global_train.call_args.kwargs['epochs'], SHORT_TRAINING['epochs'])
        flow = sp.flow_from_chain(self.chain, **self.kwargs)
        self.assertTrue(flow.is_trained)
        self.assertEqual(len(flow.population_logs), 1)
        self.assertEqual(os.listdir(self.folder), [])

    def test_flow_from_chain_with_cache(self):
        """The cache is written at the first call and loaded at the second."""
        cache_file = os.path.join(self.folder, 'flow.pt')
        flow = sp.flow_from_chain(self.chain, cache_file=cache_file, **self.kwargs)
        self.assertTrue(os.path.isfile(cache_file))
        self.assertIn('chain_fingerprint', flow.cache_record)
        with patch.object(sp.FlowCallback, 'global_train', side_effect=AssertionError('retrained')):
            cached = sp.flow_from_chain(self.chain, cache_file=cache_file, **dict(self.kwargs, feedback=1))
        self.assertIsInstance(cached, sp.FlowCallback)
        self.assertIsNot(cached, flow)
        np.testing.assert_allclose(to_array(cached.log_probability(self.chain.samples[:5])),
                                   to_array(flow.log_probability(self.chain.samples[:5])), rtol=tolerance())
        np.testing.assert_array_equal(cached.test_idx, flow.test_idx)

    def test_flow_from_chain_cache_mismatch(self):
        """A cache made with other settings raises; overwrite_cache retrains."""
        cache_file = os.path.join(self.folder, 'flow.pt')
        sp.flow_from_chain(self.chain, cache_file=cache_file, **self.kwargs)
        with self.assertRaises(ValueError) as context:
            sp.flow_from_chain(self.chain, cache_file=cache_file, **dict(self.kwargs, epochs=3))
        self.assertIn('epochs', str(context.exception))
        with self.assertRaises(ValueError) as context:
            sp.flow_from_chain(make_gaussian_chain(num_samples=300, seed=5), cache_file=cache_file, **self.kwargs)
        self.assertIn('chain', str(context.exception))
        flow = sp.flow_from_chain(self.chain, cache_file=cache_file, overwrite_cache=True,
                                  **dict(self.kwargs, epochs=3))
        self.assertEqual(len(flow.log['loss']), 3)
        cached = sp.flow_from_chain(self.chain, cache_file=cache_file, **dict(self.kwargs, epochs=3))
        self.assertEqual(len(cached.log['loss']), 3)

    def test_removed_cache_arguments_raise(self):
        """``cache_dir`` and ``root_name`` have been replaced by ``cache_file``."""
        for key in ['cache_dir', 'root_name']:
            options = dict(self.kwargs)
            options[key] = self.folder
            with self.assertRaises(ValueError):
                sp.flow_from_chain(self.chain, **options)
            with self.assertRaises(ValueError):
                sp.average_flow_from_chain(self.chain, num_flows=2, **options)

    def test_average_flow_from_chain_single_flow(self):
        """With one flow the flow itself is returned."""
        flow = sp.average_flow_from_chain(self.chain, num_flows=1, **self.kwargs)
        self.assertIsInstance(flow, sp.FlowCallback)
        self.assertNotIsInstance(flow, sp.average_flow)
        self.assertTrue(flow.is_trained)

    def test_average_flow_from_chain_without_cache(self):
        """The members share the split and are averaged."""
        with quiet():
            average = sp.average_flow_from_chain(self.chain, num_flows=2, **dict(self.kwargs, feedback=1))
        self.assertIsInstance(average, sp.average_flow)
        self.assertEqual(average.num_flows, 2)
        np.testing.assert_array_equal(average.flows[0].test_idx, average.flows[1].test_idx)
        np.testing.assert_array_equal(average.test_idx, average.flows[0].test_idx)
        self.assertEqual(len(average.test_idx), int(0.2 * 300))
        self.assertTrue(all(_f.feedback == 0 for _f in average.flows))
        self.assertEqual(os.listdir(self.folder), [])

    def test_average_flow_from_chain_with_cache(self):
        """The assembled flow is cached and the temporary files are removed."""
        cache_file = os.path.join(self.folder, 'average.pt')
        average = sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=cache_file, **self.kwargs)
        self.assertEqual(os.listdir(self.folder), ['average.pt'])
        with patch.object(sp.FlowCallback, 'global_train', side_effect=AssertionError('retrained')):
            cached = sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=cache_file, **self.kwargs)
        self.assertIsInstance(cached, sp.average_flow)
        np.testing.assert_allclose(to_array(cached.weights), to_array(average.weights), rtol=tolerance())
        np.testing.assert_allclose(to_array(cached.log_probability(self.chain.samples[:5])),
                                   to_array(average.log_probability(self.chain.samples[:5])), rtol=tolerance())
        with self.assertRaises(ValueError):
            sp.average_flow_from_chain(self.chain, num_flows=3, cache_file=cache_file, **self.kwargs)

    def test_average_flow_from_chain_mpi_needs_cache(self):
        """MPI without a cache file is disabled with a warning."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            average = sp.average_flow_from_chain(self.chain, num_flows=2, use_mpi=True, **self.kwargs)
        self.assertIn('Disabling MPI', output.getvalue())
        self.assertIsInstance(average, sp.average_flow)

    def test_average_flow_from_chain_explicit_split(self):
        """An explicit split is used by all members."""
        split = (np.arange(50), np.arange(50, 300))
        average = sp.average_flow_from_chain(self.chain, num_flows=2, validation_training_idx=split, **self.kwargs)
        for flow in average.flows:
            np.testing.assert_array_equal(flow.test_idx, split[0])
            np.testing.assert_array_equal(flow.training_idx, split[1])

    def test_average_flow_from_chain_feedback_none(self):
        """``feedback=None`` is treated as no feedback."""
        output = io.StringIO()
        with patch.object(sp.FlowCallback, 'global_train', autospec=True) as global_train, \
                contextlib.redirect_stdout(output):
            flow = sp.average_flow_from_chain(self.chain, num_flows=1, **dict(self.kwargs, feedback=None))
        global_train.assert_called_once()
        self.assertEqual(flow.feedback, 0)
        self.assertEqual(output.getvalue(), '')

    def test_cache_without_record_raises(self):
        """A snapshot saved by hand has no cache record and is not accepted as a cache."""
        cache_file = os.path.join(self.folder, 'flow.pt')
        build_flow(self.chain, trainable_bijector=None).save(cache_file)
        with self.assertRaises(ValueError) as context:
            sp.flow_from_chain(self.chain, cache_file=cache_file, **self.kwargs)
        self.assertIn('has no cache record', str(context.exception))

    def test_normalize_cache_value(self):
        """Creation arguments are converted to plain comparable values."""
        self.assertEqual(sp._normalize_cache_value(np.float32(1.5)), 1.5)
        self.assertIsInstance(sp._normalize_cache_value(np.int64(3)), int)
        self.assertEqual(sp._normalize_cache_value({'a': np.int64(2), 3: (1, None)}), {'a': 2, '3': [1, None]})
        self.assertEqual(sp._normalize_cache_value(sum_difference_forward),
                         'callable:' + __name__ + '.sum_difference_forward')
        self.assertEqual(sp._normalize_cache_value(object()), 'object:builtins.object')
        self.assertEqual(sp._normalize_cache_value(np.random.default_rng(0)),
                         sp._normalize_cache_value(np.random.default_rng(1)))
        array_value = sp._normalize_cache_value(np.arange(3.0))
        self.assertEqual(array_value['array_shape'], [3])
        self.assertNotEqual(array_value, sp._normalize_cache_value(np.arange(1.0, 4.0)))

#########################################################################################################
# Basic persistence and devices


class TestSaveLoadBasic(unittest.TestCase):
    """Basic save / load round trip and device moves."""

    def setUp(self):
        """Seed the random generators."""
        seed_everything(0)

    def test_save_and_load(self):
        """A saved flow loads without the chain and evaluates identically."""
        flow = build_trained_flow()
        points = flow.chain_samples[:5]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'flow.pt')
            flow.save(path)
            loaded = sp.FlowCallback.load(path)
        self.assertIsInstance(loaded, sp.FlowCallback)
        self.assertEqual(loaded.param_names, flow.param_names)
        self.assertEqual(loaded.name_tag, flow.name_tag)
        self.assertEqual(loaded.log, flow.log)
        self.assertTrue(loaded.is_trained)
        np.testing.assert_allclose(to_array(loaded.log_probability(points)), to_array(flow.log_probability(points)),
                                   rtol=tolerance())
        self.assertEqual(loaded.training_dataset[0].device, loaded.device)

    def test_save_invalid_mode_raises(self):
        """Unknown save modes raise ValueError."""
        flow = build_flow(trainable_bijector=None)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                flow.save(os.path.join(folder, 'flow.pt'), mode='compact')

    def test_to_same_device(self):
        """to returns the flow and keeps it usable."""
        flow = build_trained_flow()
        points = flow.chain_samples[:3]
        before = to_array(flow.log_probability(points))
        self.assertIs(flow.to(flow.device), flow)
        np.testing.assert_allclose(to_array(flow.log_probability(points)), before, rtol=tolerance())
        flow.train(**SHORT_TRAINING)

#########################################################################################################
# Run the tests


if __name__ == '__main__':
    unittest.main(verbosity=2)
