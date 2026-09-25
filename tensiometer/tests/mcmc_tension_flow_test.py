"""Tests for flow-based MCMC tension estimators."""

#########################################################################################################
# Imports

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from getdist import loadMCSamples

import tensiometer.mcmc_tension.flow as flow_mod
import tensiometer.mcmc_tension.param_diff as pd
from tensiometer.synthetic_probability import synthetic_probability as sp
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Test configuration

torch.manual_seed(0)
np.random.seed(0)

#########################################################################################################
# Helper functions


def _load_thinned_chain(path):
    """Load a test chain and thin it by its independent-sample spacing.

    :param path: root of the chain files.
    :returns: :class:`~getdist.mcsamples.MCSamples`.
    """
    chain = loadMCSamples(path)
    chain.getConvergeTests()
    chain.weighted_thin(int(chain.indep_thin))
    return chain


#########################################################################################################
# Test cases


class TestMcmcTensionFlow(unittest.TestCase):

    """MCMC tension flow test suite."""
    @classmethod
    def setUpClass(cls):
        """Load the test chains once for the test case."""
        here = os.path.dirname(os.path.abspath(__file__))
        cls.chain_1 = _load_thinned_chain(here + "/../../test_chains/DES")
        cls.chain_2 = _load_thinned_chain(here + "/../../test_chains/Planck18TTTEEE")
        cls.chain_12 = _load_thinned_chain(here + "/../../test_chains/Planck18TTTEEE_DES")
        cls.chain_prior = _load_thinned_chain(here + "/../../test_chains/prior")
        cls.diff_chain = pd.parameter_diff_chain(cls.chain_1, cls.chain_2, boost=1)

    def setUp(self):
        """Seed the random generators before each test."""
        torch.manual_seed(0)
        np.random.seed(0)

    def test_flow_runs(self):
        """Test flow estimation and short training runs."""
        class DummyFlow:
            """Dummy Flow test suite."""
            def __init__(self, chain_samples):
                """Init."""
                self.num_params = chain_samples.shape[1]
                self.chain_samples = chain_samples.astype(tu.np_prec)
                self.cov = np.eye(self.num_params)
                self.inv_cov = np.linalg.inv(self.cov)
                self.norm_const = -0.5 * np.log(np.linalg.det(2 * np.pi * self.cov))

            def log_probability(self, x):
                """Log probability."""
                x = np.array(x, dtype=tu.np_prec)
                diff = x[..., : self.num_params]
                expo = -0.5 * np.einsum("...i,ij,...j->...", diff, self.inv_cov, diff)
                return self.norm_const + expo

            def sample(self, n):
                """Sample."""
                return np.random.multivariate_normal(np.zeros(self.num_params), self.cov, size=n).astype(tu.np_prec)

            def cast(self, arr):
                """Cast."""
                return np.array(arr, dtype=tu.np_prec)

        dummy_flow = DummyFlow(self.diff_chain.samples)
        p, low, up = flow_mod.estimate_shift(dummy_flow, tol=0.5, max_iter=1, step=1000)
        self.assertGreaterEqual(p, 0.0)
        p2, low2, up2 = flow_mod.estimate_shift_from_samples(dummy_flow)
        self.assertGreaterEqual(p2, 0.0)
        res, trained_flow = flow_mod.flow_parameter_shift(self.diff_chain, epochs=1, pop_size=1, feedback=0)
        self.assertEqual(len(res), 3)
        self.assertIsInstance(trained_flow, sp.FlowCallback)
        prob, low3, up3 = res
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertLessEqual(low3, up3)

    def test_flow_parameter_shift_on_torch_flow(self):
        """Test that the shift estimators accept the tensors returned by a real flow."""
        flow = sp.FlowCallback(self.diff_chain, feedback=0, trainable_bijector=None)
        prob, low, up = flow_mod.estimate_shift(flow, tol=0.5, max_iter=1, step=500)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(low, up)
        prob2, low2, up2 = flow_mod.estimate_shift_from_samples(flow)
        self.assertGreaterEqual(prob2, 0.0)
        self.assertLessEqual(low2, up2)

    def test_flow_parameter_shift_cache(self):
        """Test the cache round trip of flow_parameter_shift."""
        options = {"epochs": 1, "pop_size": 1, "feedback": 0, "tol": 0.5, "max_iter": 1, "step": 500}
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = os.path.join(tmp_dir, "diff_flow.pt")
            res, trained_flow = flow_mod.flow_parameter_shift(self.diff_chain, cache_file=cache_file, **options)
            self.assertTrue(os.path.isfile(cache_file))
            self.assertEqual(len(res), 3)
            # the second call loads the cache without training:
            with patch.object(sp.FlowCallback, "global_train",
                              side_effect=AssertionError("the cached flow should not be trained")):
                res_2, loaded_flow = flow_mod.flow_parameter_shift(self.diff_chain, cache_file=cache_file, **options)
            self.assertEqual(len(res_2), 3)
            coords = self.diff_chain.samples[:10, :trained_flow.num_params]
            np.testing.assert_allclose(tu.to_numpy(loaded_flow.log_probability(coords)),
                                       tu.to_numpy(trained_flow.log_probability(coords)),
                                       rtol=1e-6, atol=1e-6)
            # a different chain is rejected:
            other_chain = copy.deepcopy(self.diff_chain)
            other_chain.samples[:, 0] = other_chain.samples[:, 0] + 0.1
            with self.assertRaises(ValueError):
                flow_mod.flow_parameter_shift(other_chain, cache_file=cache_file, **options)

    def test_removed_cache_arguments_raise(self):
        """Test that the removed cache_dir and root_name arguments raise."""
        with self.assertRaises(ValueError):
            flow_mod.flow_parameter_shift(self.diff_chain, cache_dir="unused", epochs=1, pop_size=1, feedback=0)
        with self.assertRaises(ValueError):
            flow_mod.flow_parameter_shift(self.diff_chain, root_name="unused", epochs=1, pop_size=1, feedback=0)

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
