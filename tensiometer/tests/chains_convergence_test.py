"""Tests for chain convergence diagnostics."""

#########################################################################################################
# Imports

import os
import unittest

import numpy as np
from getdist import MCSamples, loadMCSamples

import tensiometer.chains_convergence as conv
import tensiometer.utilities.stats_utilities as stut

#########################################################################################################


#########################################################################################################
# Convergence tests


class TestConvergence(unittest.TestCase):

    """Chain convergence diagnostics test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.chain = loadMCSamples(self.here + "/../../test_chains/DES")

    def test_gr_test_consistency(self):
        """Test Gelman-Rubin consistency for multiple chains."""
        res1 = conv.GR_test(self.chain)
        res2 = conv.GR_test(stut.get_separate_mcsamples(self.chain))
        assert np.allclose(res1[0], res2[0]) and np.allclose(res1[1], res2[1])
        res3 = conv.GR_test(
            stut.get_separate_mcsamples(self.chain),
            param_names=self.chain.getParamNames().getRunningNames(),
        )
        assert np.allclose(res1[0], res3[0]) and np.allclose(res1[1], res3[1])

    def test_gr_test_two_chains_consistency(self):
        """Test Gelman-Rubin consistency for two chains."""
        res2 = conv.GR_test(stut.get_separate_mcsamples(self.chain)[:2])
        res3 = conv.GR_test(
            stut.get_separate_mcsamples(self.chain)[:2],
            param_names=self.chain.getParamNames().getRunningNames(),
        )
        assert np.allclose(res2[0], res3[0]) and np.allclose(res2[1], res3[1])

    def test_grn_test_variants(self):
        """Test higher-moment Gelman-Rubin variants."""
        kwargs = {}
        print(conv.GRn_test(self.chain, n=2, param_names=None, feedback=2,
                            optimizer="ParticleSwarm", **kwargs))
        print(conv.GRn_test(self.chain, n=3, param_names=None, feedback=2,
                            optimizer="ParticleSwarm", **kwargs))
        print(conv.GRn_test(self.chain, n=2, param_names=None, feedback=2,
                            optimizer="TrustRegions", **kwargs))

    def test_grn_test_two_chains(self):
        """Test higher-moment Gelman-Rubin for two chains."""
        kwargs = {}
        print(conv.GRn_test(stut.get_separate_mcsamples(self.chain)[:2], n=2, param_names=None, feedback=0,
                            optimizer="ParticleSwarm", **kwargs))

    def test_validation_errors(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(TypeError):
            conv._helper_chains_to_chainlist(["not a chain"])
        with self.assertRaises(TypeError):
            conv._helper_chains_to_chainlist("not a chain container")
        with self.assertRaises(ValueError):
            conv._helper_chains_to_chainlist([stut.get_separate_mcsamples(self.chain)[0]])
        with self.assertRaises(ValueError):
            conv._helper_chains_to_chainlist([self.chain])
        with self.assertRaises(ValueError):
            conv.GRn_test_1D(self.chain, 2, param_name=['omegabh2', 'thetaMC'])
        with self.assertRaises(ValueError):
            conv.GRn_test_1D([self.chain], 2, param_name=['omegabh2'])

    def test_grn_theta0_branches(self):
        """Test theta0 branches in Gelman-Rubin n variants."""
        conv.GRn_test_1D(self.chain, 2, param_name="omegabh2", theta0=0.1)
        conv.GRn_test_1D_samples(
            [self.chain.samples[:, self.chain.index["omegabh2"]]] * 2,
            [self.chain.weights] * 2,
            3,
            theta0=0.0,
        )
        # n == 1 branch
        conv.GRn_test(self.chain, n=1)
        # single-parameter branch
        conv.GRn_test(self.chain, n=2, param_names=["omegabh2"])

    def test_gr_test_scalar_branch(self):
        """Test Gelman-Rubin scalar-parameter branch."""
        samples = [np.array([[0.0], [1.0]]), np.array([[0.5], [1.5]])]
        weights = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        res, mode = conv.GR_test_from_samples(samples, weights)
        self.assertTrue(np.isfinite(res))
        self.assertEqual(mode.shape[0], 1)

    def test_grn_param_name_length_error(self):
        """Test parameter name length validation in GRn helpers."""
        samples = np.array([[0.0, 1.0], [1.0, 2.0]])
        names = ["x", "y"]
        labels = ["x", "y"]
        chain1 = MCSamples(samples=samples, names=names, labels=labels)
        chain2 = chain1.copy()
        with self.assertRaises(ValueError):
            conv.GRn_test_1D([chain1, chain2], 2, param_name=["x", "y"])

    def test_grn_from_samples_theta0_and_feedback(self):
        """Test GRn samples path with theta0 and feedback."""
        samples = [np.array([[0.0, 1.0]]), np.array([[1.0, 2.0]])]
        weights = [np.array([1.0]), np.array([1.0])]
        original = conv.teig.max_GtRq_brute

        def _stub_max_GtRq_brute(VM, MV, feedback=0, optimizer=None, **kwargs):
            """Stub brute optimizer output for GRn tests."""
            return ("stub", VM.shape, MV.shape)

        conv.teig.max_GtRq_brute = _stub_max_GtRq_brute
        try:
            result = conv.GRn_test_from_samples(
                samples, weights, n=2, theta0=np.array([0.5, 0.5]), feedback=1, optimizer="ParticleSwarm"
            )
            self.assertIsNotNone(result)
        finally:
            conv.teig.max_GtRq_brute = original

    def test_grn_from_samples_geap_branch(self):
        """Test GRn samples path with GEAP optimizer."""
        samples = [np.array([[0.0]]), np.array([[1.0]])]
        weights = [np.array([1.0]), np.array([1.0])]
        original = conv.teig.max_GtRq_geap_power

        def _stub_geap(VM, MV, **kwargs):
            """Stub GEAP optimizer output for GRn tests."""
            return ("geap", VM.shape, MV.shape)

        conv.teig.max_GtRq_geap_power = _stub_geap
        try:
            result = conv.GRn_test_from_samples(
                samples, weights, n=2, theta0=None, feedback=0, optimizer="GEAP"
            )
            self.assertIsNotNone(result)
        finally:
            conv.teig.max_GtRq_geap_power = original


#########################################################################################################
# Slow convergence tests


class TestChainsConvergenceSlow(unittest.TestCase):
    """Chain convergence slow test suite."""
    def setUp(self):
        """Set up test fixtures."""
        here = os.path.dirname(os.path.abspath(__file__))
        self.chain = loadMCSamples(here + "/../../test_chains/DES")

    def test_grn_test_geap(self):
        """Test GEAP optimizer for GRn convergence diagnostics."""
        subchains = stut.get_separate_mcsamples(self.chain)[:2]
        res = conv.GRn_test(subchains, n=2, param_names=None, feedback=0, optimizer="GEAP")
        self.assertIsNotNone(res)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
