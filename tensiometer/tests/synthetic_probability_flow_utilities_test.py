"""Tests for flow utility helpers."""

#########################################################################################################
# Imports

import unittest

import numpy as np
import tensorflow as tf
from getdist import MCSamples

from tensiometer.synthetic_probability import flow_utilities as fu

#########################################################################################################
# Helper stubs


class DummyBijector:
    """Dummy bijector stub."""
    def __init__(self, name):
        """Init."""
        self.name = name

    def inverse(self, x):
        """Inverse."""
        return tf.convert_to_tensor(x)


class DummyFlow:
    """Dummy flow stub."""
    def __init__(self):
        """Init."""
        self.chain_samples = np.arange(12, dtype=np.float32).reshape(6, 2)
        self.chain_weights = np.ones(6, dtype=np.float32)
        self.training_idx = np.array([0, 1, 2])
        self.test_idx = np.array([3, 4, 5])
        self.training_samples = self.chain_samples[self.training_idx]
        self.training_weights = self.chain_weights[self.training_idx]
        self.test_samples = self.chain_samples[self.test_idx]
        self.test_weights = self.chain_weights[self.test_idx]
        self.trainable_bijector = type("TB", (), {"_bijectors": [DummyBijector("id1"), DummyBijector("id2")]})()
        self.fixed_bijector = DummyBijector("fixed")


class DummyProbFlow:
    """Dummy probabilistic flow stub."""
    def __init__(self, dim):
        """Init."""
        self.dim = dim

    def sample(self, n):
        """Sample."""
        return np.zeros((n, self.dim), dtype=np.float32)

    def log_probability(self, x):
        """Log probability."""
        return np.zeros(x.shape[0], dtype=np.float32)


class TestFlowUtilities(unittest.TestCase):
    """Flow utilities test suite."""
    def test_get_samples_bijectors(self):
        """Test get_samples_bijectors output."""
        flow = DummyFlow()
        train_samples, val_samples = fu.get_samples_bijectors(flow, feedback=False)
        self.assertIsInstance(train_samples[0], MCSamples)
        self.assertEqual(len(train_samples), len(flow.trainable_bijector._bijectors) + 2)
        fu.get_samples_bijectors(flow, feedback=True)
        extra = np.ones((2, 2), dtype=np.float32)
        train_samples, val_samples, extra_samples = fu.get_samples_bijectors(flow, feedback=False, extra_samples=extra)
        self.assertEqual(len(extra_samples), len(flow.trainable_bijector._bijectors) + 2)

    def test_kl_divergence(self):
        """Test KL divergence helper."""
        f1 = DummyProbFlow(2)
        f2 = DummyProbFlow(2)
        mean, std = fu.KL_divergence(f1, f2, num_samples=10, num_batches=2)
        self.assertEqual(mean, 0.0)
        self.assertEqual(std, 0.0)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
