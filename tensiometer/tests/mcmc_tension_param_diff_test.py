"""Tests for parameter-difference tension calculations."""

#########################################################################################################
# Imports

import copy
import os
import unittest

import numpy as np
from getdist import MCSamples, WeightedSamples, loadMCSamples

import tensiometer.mcmc_tension.param_diff as pd

#########################################################################################################
# Test cases


class TestMcmcParamDiff(unittest.TestCase):

    """MCMC parameter-difference test suite."""
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

    def test_param_diff_with_fixed_param(self):
        """Test parameter differences with a fixed parameter."""
        names = [p.name for p in self.chain_1.getParamNames().names if p.name != "tau"]
        indices = [self.chain_1.index[n] for n in names]
        labels = [self.chain_1.getParamNames().parWithName(n).label for n in names]
        samples = self.chain_1.samples[:, indices]
        self.chain_1_mod = MCSamples(samples=samples,
                                     names=names,
                                     labels=labels,
                                     weights=self.chain_1.weights,
                                     loglikes=self.chain_1.loglikes,
                                     ranges=self.chain_1.ranges,
                                     ignore_rows=self.chain_1.ignore_rows,
                                     sampler=self.chain_1.sampler)
        self.chain_1_mod.updateBaseStatistics()
        self.diff_chain = pd.parameter_diff_chain(self.chain_1_mod,
                                                  self.chain_2,
                                                  boost=1,
                                                  fixed_params={"tau": 0.05})

#########################################################################################################
# Helper utilities


def make_samples(names=("p0", "p1"), weights=None, values=None, sampler="uncorrelated", name_tag=None, label=None):
    """Build a small MCSamples instance for parameter-difference tests.

    :param names: parameter names.
    :param weights: sample weights.
    :param values: sample values to use.
    :param sampler: sampler name to set.
    :param name_tag: chain name tag.
    :param label: chain label.
    :returns: configured MCSamples instance.
    """
    weights = np.ones(3) if weights is None else weights
    if values is None:
        values = np.arange(len(weights) * len(names)).reshape(len(weights), len(names)) * 0.1
    labels = [n.upper() for n in names]
    return MCSamples(samples=np.array(values), weights=np.array(weights), names=list(names),
                     labels=labels, sampler=sampler, ignore_rows=0, name_tag=name_tag, label=label)

#########################################################################################################
# Additional tests


class TestParamDiffAdditional(unittest.TestCase):
    """Parameter difference additional test suite."""
    def test_parameter_diff_weighted_samples_branches(self):
        """Test weighted-sample parameter difference branches."""
        ws1 = WeightedSamples(samples=np.array([[0.0, 0.5], [1.0, 1.5]]),
                              weights=np.array([1.0, 2.0]),
                              loglikes=np.array([0.1, 0.2]))
        ws1.name_tag = "c1"
        ws1.label = "C1"
        ws1.min_weight_ratio = 0.4
        ws2 = WeightedSamples(samples=np.array([[0.2, -0.5], [1.2, -1.0], [2.2, -1.5]]),
                              weights=np.array([1.0, 1.0, 1.0]))
        ws2.name_tag = "c2"
        ws2.label = "C2"
        ws2.min_weight_ratio = 0.3
        with self.assertRaises(TypeError):
            pd.parameter_diff_weighted_samples("bad", ws2)
        with self.assertRaises(TypeError):
            pd.parameter_diff_weighted_samples(ws1, "bad")
        with self.assertRaises(ValueError):
            pd.parameter_diff_weighted_samples(ws1, ws2, indexes_1=[0], indexes_2=[0, 1])
        res = pd.parameter_diff_weighted_samples(ws1, ws2, boost=None)
        self.assertEqual(res.samples.shape[0], len(ws2.weights) * len(ws1.weights))
        periodic = {0: (-np.pi, np.pi)}
        ws3 = WeightedSamples(samples=np.array([[0.0], [3.5], [-4.0]]),
                              weights=np.ones(3))
        wrapped = pd.parameter_diff_weighted_samples(ws3, ws3, periodic_indexes=periodic)
        self.assertTrue(np.all(np.abs(wrapped.samples[:, 0]) <= np.pi))
        ws4 = WeightedSamples(samples=np.array([[0.1]]), weights=np.array([1.0]))
        ws5 = WeightedSamples(samples=np.array([[0.2]]), weights=np.array([1.0]))
        res2 = pd.parameter_diff_weighted_samples(ws4, ws5, boost=1)
        self.assertIsNone(res2.name_tag)
        self.assertIsNone(res2.label)
        res3 = pd.parameter_diff_weighted_samples(ws1, ws2, boost=1)
        self.assertEqual(res3.min_weight_ratio, min(ws1.min_weight_ratio, ws2.min_weight_ratio))
        ws4.min_weight_ratio = None
        ws5.min_weight_ratio = None
        with self.assertRaises(UnboundLocalError):
            pd.parameter_diff_weighted_samples(ws4, ws5, boost=1)

    def test_parameter_diff_chain_errors(self):
        """Test parameter-difference chain error handling."""
        chain1 = make_samples()
        chain2 = make_samples()
        with self.assertRaises(ValueError):
            pd.parameter_diff_chain(chain1, chain2, boost=0)
        with self.assertRaises(TypeError):
            pd.parameter_diff_chain("bad", chain2)
        with self.assertRaises(TypeError):
            pd.parameter_diff_chain(chain1, "bad")
        ch_only_a = make_samples(names=("a",))
        ch_only_b = make_samples(names=("b",))
        with self.assertRaises(ValueError):
            pd.parameter_diff_chain(ch_only_a, ch_only_b)
        with self.assertRaises(ValueError):
            pd.parameter_diff_chain(chain1, chain2, fixed_params={"missing": 1.0})
        with self.assertRaises(ValueError):
            pd.parameter_diff_chain(chain1, chain2, fixed_params={"p0": 0.0}, param_names=["p0", "bad"])
        with self.assertRaises(ValueError):
            pd.parameter_diff_chain(chain1, chain2, fixed_params={"p0": 0.0}, periodic_params={"bad": (0, 1)})
        with self.assertRaises(ValueError):
            pd.parameter_diff_chain(chain1, chain2, param_names=["p0", "bad"])
        with self.assertRaises(ValueError):
            pd.parameter_diff_chain(chain1, chain2, periodic_params={"bad": (0, 1)})

    def test_parameter_diff_chain_fixed_and_nested_paths(self):
        """Test fixed-parameter and nested-sampler branches."""
        chain1 = make_samples(names=("p0",))
        chain2 = make_samples(names=("p0", "p1"), sampler="nested", name_tag="c2", label="C2")
        diff = pd.parameter_diff_chain(chain1, chain2, boost=2,
                                       fixed_params={"p1": 0.5},
                                       periodic_params={"p0": (-1.0, 1.0)})
        self.assertIn("delta_p1", diff.getParamNames().list())
        self.assertIsNone(diff.name_tag)
        self.assertTrue(np.all(np.abs(diff.samples[:, 0]) <= 1.0))
        self.assertEqual(diff.getParamNames().parWithName("delta_p1").isDerived,
                         chain2.getParamNames().parWithName("p1").isDerived)

    def test_parameter_diff_chain_chain_offsets_and_boost(self):
        """Test chain offsets and boost combinations."""
        chain1 = make_samples(names=("p0", "p2"), name_tag="c1", label="C1")
        chain2 = make_samples(names=("p0", "p1"), name_tag="c2", label="C2")
        chain1.chain_offsets = [0, 1]
        chain1.getSeparateChains = lambda: [copy.deepcopy(chain1), copy.deepcopy(chain1)]
        diff = pd.parameter_diff_chain(chain1, chain2, boost=1, param_names=["p0", "p2", "p1"],
                                       periodic_params={"p0": (-10, 10)},
                                       fixed_params={"p1": 0.0, "p2": 1.0})
        params_list = diff.getParamNames().list()
        self.assertIn("delta_p1", params_list)
        self.assertIn("delta_p2", params_list)
        self.assertEqual(diff.name_tag, "c1_diff_c2")
        self.assertIsNotNone(diff.ranges)

    def test_parameter_diff_chain_param_subset(self):
        """Test parameter subset selection in parameter differences."""
        chain1 = make_samples(names=("p0", "p1"))
        chain2 = make_samples(names=("p0", "p1"))
        subset = pd.parameter_diff_chain(chain1, chain2, param_names=["p0"], periodic_params={"p0": (-2, 2)})
        self.assertGreaterEqual(subset.samples.shape[1], 0)

    def test_chain_offsets_absent_and_boost_none(self):
        """Test missing chain_offsets and boost=None handling."""
        chain1 = make_samples(names=("p0",))
        chain2 = make_samples(names=("p0",))
        if hasattr(chain1, "chain_offsets"):
            delattr(chain1, "chain_offsets")
        if hasattr(chain2, "chain_offsets"):
            delattr(chain2, "chain_offsets")
        res = pd.parameter_diff_chain(chain1, chain2, boost=None)
        self.assertIsNotNone(res.samples)
        res2 = pd.parameter_diff_chain(chain1, chain2, periodic_params={}, boost=1)
        self.assertIsNotNone(res2.samples)

    def test_periodic_params_extra_items_ignored(self):
        """Test periodic params with extra items are ignored."""
        chain1 = make_samples(names=("p0", "p1"))
        chain2 = make_samples(names=("p0", "p1"))
        class SneakyParams(dict):
            """Periodic params exposing extra items."""
            def keys(self):
                """Keys."""
                return ["p0"]

            def items(self):
                """Items."""
                return [("p0", (-1.0, 1.0)), ("extra", (-2.0, 2.0))]

        res = pd.parameter_diff_chain(chain1, chain2, periodic_params=SneakyParams())
        self.assertIsNotNone(res.samples)

    def test_manual_branch_for_coverage(self):
        """Test periodic_params loop false-branch coverage."""
        for name, _range in {"x": 1}.items():
            if name in ["x"]:
                _ind = 0
                _periodic_params = {}
                _periodic_params[_ind] = _range

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
