"""Tests for flow-based MCMC tension estimators."""

#########################################################################################################
# Imports

import os
import unittest

import numpy as np
import tensorflow as tf
from getdist import loadMCSamples

import tensiometer.mcmc_tension.flow as flow_mod
import tensiometer.mcmc_tension.param_diff as pd

#########################################################################################################
# Test configuration

tf.random.set_seed(0)
np.random.seed(0)

#########################################################################################################
# Test cases


class TestMcmcTensionFlow(unittest.TestCase):

    """MCMC tension flow test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.chain_1 = loadMCSamples(self.here + "/../../test_chains/DES")
        self.chain_2 = loadMCSamples(self.here + "/../../test_chains/Planck18TTTEEE")
        self.chain_12 = loadMCSamples(self.here + "/../../test_chains/Planck18TTTEEE_DES")
        self.chain_prior = loadMCSamples(self.here + "/../../test_chains/prior")
        self.chain_1.getConvergeTests()
        self.chain_2.getConvergeTests()
        self.chain_12.getConvergeTests()
        self.chain_prior.getConvergeTests()
        self.chain_1.weighted_thin(int(self.chain_1.indep_thin))
        self.chain_2.weighted_thin(int(self.chain_2.indep_thin))
        self.chain_12.weighted_thin(int(self.chain_12.indep_thin))
        self.chain_prior.weighted_thin(int(self.chain_prior.indep_thin))
        self.diff_chain = pd.parameter_diff_chain(self.chain_1, self.chain_2, boost=1)

    def test_flow_runs(self):
        """Test flow estimation and short training runs."""
        class DummyFlow:
            """Dummy Flow test suite."""
            def __init__(self, chain_samples):
                """Init."""
                self.num_params = chain_samples.shape[1]
                self.chain_samples = chain_samples.astype(np.float32)
                self.cov = np.eye(self.num_params)
                self.inv_cov = np.linalg.inv(self.cov)
                self.norm_const = -0.5 * np.log(np.linalg.det(2 * np.pi * self.cov))

            def log_probability(self, x):
                """Log probability."""
                x = np.array(x, dtype=np.float32)
                diff = x[..., : self.num_params]
                expo = -0.5 * np.einsum("...i,ij,...j->...", diff, self.inv_cov, diff)
                return self.norm_const + expo

            def sample(self, n):
                """Sample."""
                return np.random.multivariate_normal(np.zeros(self.num_params), self.cov, size=n).astype(np.float32)

            def cast(self, arr):
                """Cast."""
                return np.array(arr, dtype=np.float32)

        dummy_flow = DummyFlow(self.diff_chain.samples)
        p, low, up = flow_mod.estimate_shift(dummy_flow, tol=0.5, max_iter=1, step=1000)
        self.assertGreaterEqual(p, 0.0)
        p2, low2, up2 = flow_mod.estimate_shift_from_samples(dummy_flow)
        self.assertGreaterEqual(p2, 0.0)
        res, trained_flow = flow_mod.flow_parameter_shift(self.diff_chain, epochs=1, pop_size=1, feedback=0)
        self.assertEqual(len(res), 3)

    def test_estimate_shift_with_prior_flow(self):
        """Test estimate_shift with prior flow inputs."""
        class SimpleFlow:
            """Simple Flow test suite."""
            def __init__(self, num_params=1, chain_samples=None, log_val=0.0, scale=0.1):
                """Init."""
                self.num_params = num_params
                self.chain_samples = chain_samples if chain_samples is not None else np.zeros((4, num_params))
                self.log_val = log_val
                self.scale = scale

            def log_probability(self, x):
                """Log probability."""
                x = np.array(x)
                return np.arange(x.shape[0], dtype=float) * self.scale + self.log_val

            def sample(self, n):
                """Sample."""
                return np.zeros((n, self.num_params))

            def cast(self, arr):
                """Cast."""
                return np.array(arr)

        base_flow = SimpleFlow(scale=0.1)
        prior_flow = SimpleFlow(log_val=0.0, scale=-0.05)
        prob, low, high = flow_mod.estimate_shift(base_flow, prior_flow=prior_flow, tol=10.0, max_iter=0, step=5)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(low, high)

        sampled_flow = SimpleFlow(chain_samples=np.zeros((6, 1)), scale=0.2)
        prior_samples = SimpleFlow(chain_samples=np.zeros((6, 1)), log_val=-0.1, scale=0.0)
        prob2, low2, high2 = flow_mod.estimate_shift_from_samples(sampled_flow, prior_flow=prior_samples)
        self.assertGreaterEqual(prob2, 0.0)
        self.assertLessEqual(low2, high2)
        self.assertFalse(np.isnan(low))
        self.assertFalse(np.isnan(low2))


#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
