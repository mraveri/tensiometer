"""Tests for flow utility helpers."""

#########################################################################################################
# Imports

import unittest

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from getdist import MCSamples

from tensiometer.synthetic_probability import flow_utilities as fu
from tensiometer.synthetic_probability import synthetic_probability as sp
from tensiometer.synthetic_probability import tensor_utilities as tu
from tensiometer.synthetic_probability import trainable_bijectors as tb

#########################################################################################################
# Helper stubs


class DummyBijector:
    """Dummy bijector stub."""
    def __init__(self, name):
        """Init."""
        self.name = name

    def inverse(self, x):
        """Inverse."""
        return torch.as_tensor(x)


class DummyFlow:
    """Dummy flow stub."""
    def __init__(self):
        """Init."""
        self.chain_samples = np.arange(12, dtype=tu.np_prec).reshape(6, 2)
        self.chain_weights = np.ones(6, dtype=tu.np_prec)
        self.training_idx = np.array([0, 1, 2])
        self.test_idx = np.array([3, 4, 5])
        self.training_samples = self.chain_samples[self.training_idx]
        self.training_weights = self.chain_weights[self.training_idx]
        self.test_samples = self.chain_samples[self.test_idx]
        self.test_weights = self.chain_weights[self.test_idx]
        self.trainable_bijector = type("TB", (), {"bijectors": [DummyBijector("id1"), DummyBijector("id2")]})()
        self.fixed_bijector = DummyBijector("fixed")


class DummyProbFlow:
    """Dummy probabilistic flow stub returning CPU tensors like the flow API."""
    def __init__(self, dim):
        """Init."""
        self.dim = dim

    def sample(self, n):
        """Sample."""
        return torch.zeros((n, self.dim), dtype=tu.get_precision())

    def log_probability(self, x):
        """Log probability."""
        return torch.zeros(x.shape[0], dtype=tu.get_precision())


#########################################################################################################
# Helper functions


def _gaussian_chain(num_samples=300, seed=0):
    """Build a small correlated two dimensional Gaussian chain.

    :param num_samples: number of samples.
    :param seed: seed of the random generator.
    :returns: :class:`~getdist.mcsamples.MCSamples` with parameters ``x`` and ``y``.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = rng.multivariate_normal([0.0, 1.0], cov, size=num_samples)
    return MCSamples(samples=samples, names=["x", "y"], labels=["x", "y"], sampler="uncorrelated")


def _small_flow(chain):
    """Build a small untrained flow with an autoregressive trainable bijector.

    :param chain: input chain.
    :returns: :class:`~tensiometer.synthetic_probability.synthetic_probability.FlowCallback`.
    """
    trainable = tb.AutoregressiveFlow(
        2, n_transformations=2, hidden_units=[4], permutations=[[1, 0], [1, 0]])
    return sp.FlowCallback(chain, trainable_bijector=trainable, feedback=0,
                           rng=np.random.default_rng(0))


#########################################################################################################
# Test cases


class TestFlowUtilities(unittest.TestCase):
    """Flow utilities test suite."""
    def test_get_samples_bijectors(self):
        """Test get_samples_bijectors output."""
        flow = DummyFlow()
        train_samples, val_samples = fu.get_samples_bijectors(flow, feedback=False)
        self.assertIsInstance(train_samples[0], MCSamples)
        self.assertEqual(len(train_samples), len(flow.trainable_bijector.bijectors) + 2)
        fu.get_samples_bijectors(flow, feedback=True)
        extra = np.ones((2, 2), dtype=tu.np_prec)
        train_samples, val_samples, extra_samples = fu.get_samples_bijectors(flow, feedback=False, extra_samples=extra)
        self.assertEqual(len(extra_samples), len(flow.trainable_bijector.bijectors) + 2)
        for samples in extra_samples[1:]:
            self.assertIsInstance(samples, np.ndarray)

    def test_kl_divergence(self):
        """Test KL divergence helper."""
        f1 = DummyProbFlow(2)
        f2 = DummyProbFlow(2)
        mean, std = fu.KL_divergence(f1, f2, num_samples=10, num_batches=2)
        self.assertEqual(mean, 0.0)
        self.assertEqual(std, 0.0)


class TestFlowUtilitiesRealFlow(unittest.TestCase):
    """Flow utilities on a small real flow."""
    @classmethod
    def setUpClass(cls):
        """Build the flow once for the test case."""
        torch.manual_seed(0)
        np.random.seed(0)
        cls.chain = _gaussian_chain()
        cls.flow = _small_flow(cls.chain)

    def test_get_samples_bijectors_real_flow(self):
        """Test get_samples_bijectors on a real flow."""
        flow = self.flow
        num_bijectors = len(flow.trainable_bijector.bijectors)
        self.assertEqual(num_bijectors, 4)
        extra = self.chain.samples[:5]
        train_samples, val_samples, extra_samples = fu.get_samples_bijectors(
            flow, feedback=False, extra_samples=extra)
        self.assertEqual(len(train_samples), num_bijectors + 2)
        self.assertEqual(len(val_samples), num_bijectors + 2)
        self.assertEqual(len(extra_samples), num_bijectors + 2)
        num_train = len(flow.training_idx)
        num_test = len(flow.test_idx)
        for samples in train_samples:
            self.assertIsInstance(samples, MCSamples)
            self.assertEqual(samples.samples.shape, (num_train, 2))
        for samples in val_samples:
            self.assertEqual(samples.samples.shape, (num_test, 2))
        for samples in extra_samples[1:]:
            self.assertIsInstance(samples, np.ndarray)
            self.assertEqual(samples.shape, (5, 2))
        # the name tags follow the bijector names:
        self.assertEqual(train_samples[-1].name_tag,
                         str(num_bijectors - 1) + "_after_" + flow.trainable_bijector.bijectors[-1].name)
        # the last space is the full inverse of the trainable bijector:
        with torch.no_grad():
            expected = tu.to_numpy(flow.trainable_bijector.inverse(
                tu.to_tensor(flow.training_samples, device=flow.device)))
        np.testing.assert_allclose(train_samples[-1].samples, expected, rtol=1e-5, atol=1e-5)
        # the extra samples go through the full bijector:
        np.testing.assert_allclose(extra_samples[-1], tu.to_numpy(flow.map_to_abstract_coord(extra)),
                                   rtol=1e-4, atol=1e-4)

    def test_kl_divergence_real_flow(self):
        """Test that the KL divergence of a flow with itself vanishes."""
        torch.manual_seed(1)
        mean, std = fu.KL_divergence(self.flow, self.flow, num_samples=50, num_batches=3)
        self.assertTrue(np.isfinite(mean))
        self.assertAlmostEqual(mean, 0.0, places=6)
        self.assertAlmostEqual(std, 0.0, places=6)


#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
