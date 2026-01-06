"""Tests for Gaussian tension utilities."""

#########################################################################################################
# Imports

import unittest
from unittest.mock import patch

import numpy as np
from getdist import MCSamples, loadMCSamples
from getdist.gaussian_mixtures import GaussianND

import tensiometer.gaussian_tension as gt

#########################################################################################################
# Shared test fixtures


def setup_test(target):
    """Set up shared Gaussian test fixtures.

    :param target: test case instance to receive attributes.
    :returns: ``None``.
    """
    target.n1 = 2
    target.n2 = 3
    target.mean1 = 1.0 * np.ones(target.n1)
    target.mean2 = 2.0 * np.ones(target.n2)
    target.cov1 = 1.0 * np.diag(np.ones(target.n1))
    target.cov2 = 2.0 * np.diag(np.ones(target.n2))
    target.param_names1 = [f"p{i}" for i in range(target.n1)]
    target.param_names2 = [f"p{i}" for i in range(target.n2)]
    target.gaussian_1 = GaussianND(target.mean1, target.cov1, names=target.param_names1)
    target.gaussian_2 = GaussianND(target.mean2, target.cov2, names=target.param_names2)
    target.chain_1 = target.gaussian_1.MCSamples(1000)
    target.chain_2 = target.gaussian_2.MCSamples(1000)
    target.chain_12 = target.gaussian_2.MCSamples(1000)
    target.gaussian_prior = GaussianND(target.mean2, 10.0 * target.cov2, names=target.param_names2)
    target.prior_chain = target.gaussian_prior.MCSamples(1000)

#########################################################################################################
# Helper tests


class TestHelpers(unittest.TestCase):

    """Gaussian tension helper test suite."""
    def setUp(self):
        """Set up test fixtures."""
        setup_test(self)

    def test_helpers(self):
        """Test helper utilities for parameter checks."""
        self.assertEqual(
            self.chain_1.getParamNames().getRunningNames(),
            gt._check_param_names(self.chain_1, param_names=None),
        )
        self.assertEqual(gt._check_param_names(self.chain_1, param_names=["p1"]), ["p1"])
        gt._check_chain_type(self.chain_1)

    def test_validation_errors(self):
        """Test error handling for invalid helper inputs."""
        with self.assertRaises(ValueError):
            gt._check_param_names(self.chain_1, param_names=["test"])
        with self.assertRaises(TypeError):
            gt._check_chain_type(self.gaussian_1)
        other_chain = GaussianND(np.zeros(1), np.eye(1), names=["x"]).MCSamples(10)
        with self.assertRaises(ValueError):
            gt.Q_DM(self.chain_1, other_chain)

#########################################################################################################
# Utility tests


class TestUtilities(unittest.TestCase):

    """Gaussian tension utility test suite."""
    def setUp(self):
        """Set up test fixtures."""
        setup_test(self)

    def test_get_prior_covariance(self):
        """Test prior covariance estimation with ranges."""
        self.chain_1.setRanges({"p0": [0.0, 1.0], "p1": [0.0, 1.0]})
        gt.get_prior_covariance(self.chain_1)
        gt.get_prior_covariance(self.chain_2)

    def test_get_neff(self):
        """Test effective sample size calculations."""
        assert np.allclose(gt.get_Neff(self.chain_1), 2.0)
        gt.get_Neff(self.chain_1, prior_chain=self.prior_chain)
        assert np.allclose(gt.get_Neff(self.chain_1, param_names=["p1"]), 1.0)
        assert np.allclose(gt.get_Neff(self.chain_1, prior_factor=1.0), 2.0)

    def test_gaussian_approximation(self):
        """Test Gaussian approximation helper."""
        gt.gaussian_approximation(self.chain_1)
        gt.gaussian_approximation(self.chain_1, param_names=["p1"])
        self.chain_1.label = "chain_1"
        temp = gt.gaussian_approximation(self.chain_1)
        assert temp.label == "Gaussian " + self.chain_1.label
        self.chain_1.label = None
        self.chain_1.name_tag = "chain_1"
        temp = gt.gaussian_approximation(self.chain_1)
        assert temp.label == "Gaussian_" + self.chain_1.name_tag

    def test_q_metrics(self):
        """Test Q_MAP and Q_DMAP metrics."""
        q_map, dof = gt.Q_MAP(self.chain_1, num_data=10)
        self.assertTrue(np.isfinite(q_map))
        q_dmap, dofs = gt.Q_DMAP(self.chain_1, self.chain_2, self.chain_12, prior_chain=self.prior_chain)
        self.assertTrue(np.isfinite(q_dmap))


#########################################################################################################
# Slow integration tests


class TestGaussianTensionSlow(unittest.TestCase):
    """Gaussian tension slow test suite."""
    def setUp(self):
        """Set up test fixtures."""
        names = ["p0", "p1"]
        self.chain_1 = gt.GaussianND(np.array([0.0, 0.0]), np.eye(2), names=names).MCSamples(500)
        self.chain_2 = gt.GaussianND(np.array([0.2, -0.1]), 1.5 * np.eye(2), names=names).MCSamples(500)
        self.chain_12 = gt.GaussianND(np.array([0.1, -0.05]), 0.8 * np.eye(2), names=names).MCSamples(500)
        self.prior_chain = gt.GaussianND(np.array([0.0, 0.0]), 5.0 * np.eye(2), names=names).MCSamples(500)

    def test_localized_neff_and_cutoff(self):
        """Test localized effective sample size and cutoff estimation."""
        neff_localized = gt.get_Neff(self.chain_1, prior_chain=self.prior_chain,
                                     param_names=["p0", "p1"], localize=True, scale=5.0)
        self.assertGreaterEqual(neff_localized, 0)
        cutoff, eig, eigv, names = gt.Q_UDM_get_cutoff(self.chain_1, self.chain_2, self.chain_12,
                                                       prior_chain=self.prior_chain,
                                                       param_names=["p0", "p1"],
                                                       prior_factor=0.5)
        self.assertTrue(cutoff >= 1.0)
        self.assertEqual(len(names), eigv.shape[0])

    def test_cpca_pipeline_and_print(self):
        """Test CPCA pipeline and printing."""
        fisher_1 = np.linalg.inv(self.chain_1.cov(pars=["p0", "p1"]))
        fisher_12 = np.linalg.inv(self.chain_12.cov(pars=["p0", "p1"]))
        res = gt.linear_CPCA(fisher_1, fisher_12, ["p0", "p1"],
                             conditional_params=["p0"],
                             marginalized_parameters=[],
                             dimensional_reduce=False,
                             normparam="p0",
                             dimensional_threshold=0.05)
        self.assertIn("CPCA_eig", res)
        chain_res = gt.linear_CPCA_chains(self.chain_1, self.chain_12, ["p0", "p1"])
        text_verbose = gt.print_CPCA_results(chain_res, verbose=True, num_modes=1)
        text_brief = gt.print_CPCA_results(chain_res, verbose=False, num_modes=1)
        self.assertIn("CPCA for", text_verbose)
        self.assertIn("CPCA for", text_brief)

    def test_q_functions(self):
        """Test Q_UDM and component helpers."""
        cutoff = gt.Q_UDM_get_cutoff(self.chain_1, self.chain_2, self.chain_12,
                                     prior_chain=self.prior_chain)[0]
        q_udm, dofs_udm = gt.Q_UDM(self.chain_1, self.chain_12, lower_cutoff=1.0,
                                   upper_cutoff=cutoff + 1.0)
        self.assertGreaterEqual(dofs_udm, 0)
        eig, eigv, names = gt.Q_UDM_KL_components(self.chain_1, self.chain_12, param_names=["p0", "p1"])
        self.assertEqual(len(names), eigv.shape[0])

#########################################################################################################
# Additional test helpers


def make_simple_chain():
    """Make a simple 2D MCSamples chain.

    :returns: chain with two parameters.
    """
    samples = np.array([[0.0, 1.0], [1.0, 2.0]])
    names = ["p0", "p1"]
    labels = ["p0", "p1"]
    chain = MCSamples(samples=samples, names=names, labels=labels)
    return chain


#########################################################################################################
# Additional tests


class TestGaussianTensionAdditional(unittest.TestCase):
    """Gaussian tension additional test suite."""
    def test_check_param_names_error(self):
        """Test parameter name validation errors."""
        chain = make_simple_chain()
        with self.assertRaises(ValueError):
            gt._check_param_names(chain, ["missing"])

    def test_check_common_names_error(self):
        """Test shared-name validation errors."""
        with self.assertRaises(ValueError):
            gt._check_common_names(["a"], ["b"])

    def test_get_prior_covariance_with_missing_ranges(self):
        """Test prior covariance with missing parameter ranges."""
        chain = make_simple_chain()
        cov = gt.get_prior_covariance(chain, param_names=["p0", "p1"])
        self.assertEqual(cov.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(cov)))

    def test_linear_cpca_errors(self):
        """Test linear CPCA validation errors."""
        fisher = np.eye(2)
        with self.assertRaises(ValueError):
            gt.linear_CPCA(fisher_1=fisher, fisher_12=np.eye(3), param_names=["p0", "p1", "p2"])
        with self.assertRaises(ValueError):
            gt.linear_CPCA(fisher_1=fisher, fisher_12=fisher, param_names=["p0", "p1"], conditional_params=["bad"])
        with self.assertRaises(ValueError):
            gt.linear_CPCA(fisher_1=fisher, fisher_12=fisher, param_names=["p0", "p1"], marginalized_parameters=["bad"])

    def test_linear_cpca_marginalization_and_normparam(self):
        """Test linear CPCA marginalization and normalization settings."""
        fisher1 = np.array([[2.0, 0.1], [0.1, 1.5]])
        fisher12 = np.array([[1.5, 0.0], [0.0, 1.0]])
        results = gt.linear_CPCA(
            fisher_1=fisher1,
            fisher_12=fisher12,
            param_names=["p0", "p1"],
            marginalized_parameters=["p1"],
            normparam="p0",
            dimensional_reduce=True,
            dimensional_threshold=0.0,
        )
        self.assertIn("CPCA_eig", results)
        self.assertIn("CPCA_projector", results)
        self.assertEqual(results["CPCA_eig"].shape[0], results["CPCA_eigv"].shape[0])

    def test_localized_covariance_param_error(self):
        """Test localized covariance parameter validation."""
        chain = make_simple_chain()
        with self.assertRaises(ValueError):
            gt.get_localized_covariance(chain, chain, param_names=["p0", "p1"], localize_params=["bad"])

    def test_q_dm_cutoff_and_prior_mismatch(self):
        """Test Q_DM cutoff and prior mismatch validation."""
        chain1 = make_simple_chain()
        chain2 = make_simple_chain()
        chain1.setRanges({"p0": [-1.0, 1.0], "p1": [-1.0, 1.0]})
        chain2.setRanges({"p0": [0.0, 1.0], "p1": [0.0, 1.0]})
        with self.assertRaises(ValueError):
            gt.Q_DM(chain1, chain2, cutoff=-0.1)
        with self.assertRaises(ValueError):
            gt.Q_DM(chain1, chain2, cutoff=0.1)

    def test_cpca_print_branches(self):
        """Test CPCA print formatting branches."""
        cpca = {
            "param_names": ["a", "b"],
            "param_labels": ["A", "B"],
            "conditional_params": ["cond"],
            "marginalized_parameters": ["marg"],
            "Neff": 1.5,
            "Neff_spectrum": np.array([1.0, 0.5]),
            "KL_divergence": 0.2,
            "KL_spectrum": np.array([0.2, 0.1]),
            "CPCA_eig": np.array([2.0, 0.5]),
            "CPCA_eigv": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "CPCA_projector": np.array([[1.0, 0.0], [0.0, 0.5]]),
            "CPCA_var_contributions": np.array([[0.6, 0.1], [0.4, 0.0]]),
            "CPCA_var_filter": np.array([[True, False], [True, False]]),
            "normparam": 0,
            "reference_point": np.array([0.1, -0.2]),
            "correlation_mode_parameter": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "correlation_parameter_names": ["c1", "c2"],
        }
        text_verbose = gt.print_CPCA_results(cpca, verbose=True, num_modes=None)
        text_brief = gt.print_CPCA_results(cpca, verbose=False, num_modes=1)
        self.assertIn("CPCA for", text_verbose)
        self.assertIn("CPCA for", text_brief)

    def test_q_udm_component_errors(self):
        """Test Q_UDM component validation errors."""
        chain = make_simple_chain()
        with self.assertRaises(ValueError):
            gt.Q_UDM_fisher_components(chain, chain, which="bad")
        with self.assertRaises(ValueError):
            gt.Q_UDM_covariance_components(chain, chain, which="bad")

    def test_get_map_loglike_paths(self):
        """Test MAP loglike extraction paths."""
        class BestFit:
            """Best Fit test suite."""
            def __init__(self):
                """Init."""
                self.chiSquareds = [("data", type("C", (), {"chisq": 2.0})())]
                self.logLike = 0.0

            def getParamDict(self):
                """Get Param Dict."""
                return {}

        class Stats:
            """Stats test suite."""
            def __init__(self):
                """Init."""
                self.logLike_sample = -1.0

            def list(self):
                """List."""
                return ["chi2_data", "chi2_prior"]

            def parWithName(self, name):
                """Par With Name."""
                return type("P", (), {"bestfit_sample": 4.0})()

        class DummyChain:
            """Dummy Chain test suite."""
            def getBestFit(self, max_posterior=True):
                """Get Best Fit."""
                return BestFit()

        class DummyChainFallback:
            """Dummy Chain Fallback test suite."""
            def getBestFit(self, max_posterior=True):
                """Get Best Fit."""
                raise RuntimeError("no best fit")

            def getLikeStats(self):
                """Get Like Stats."""
                return Stats()

        val = gt.get_MAP_loglike(DummyChain(), feedback=False)
        self.assertTrue(np.isfinite(val))
        self.assertEqual(BestFit().getParamDict(), {})
        val2 = gt.get_MAP_loglike(DummyChainFallback(), feedback=False)
        self.assertTrue(np.isfinite(val2))

#########################################################################################################
# Targeted test helpers


def make_chain(mean_offset=0.0, name_tag="c"):
    """Make a small chain with two parameters.

    :param mean_offset: offset applied to sample means.
    :param name_tag: chain name tag to attach.
    :returns: chain with ranges set for ``p0`` and ``p1``.
    """
    samples = np.array([[0.1 + mean_offset, 0.2 + mean_offset],
                        [0.4 + mean_offset, -0.3 + mean_offset],
                        [-0.2 + mean_offset, 0.5 + mean_offset]])
    chain = MCSamples(samples=samples, names=["p0", "p1"], labels=["p0", "p1"], name_tag=name_tag)
    chain.setRanges({"p0": [-1.0, 1.0], "p1": [-2.0, 2.0]})
    return chain


#########################################################################################################
# Targeted tests


class TestGaussianTensionTargeted(unittest.TestCase):
    """Gaussian tension targeted test suite."""
    def test_localized_covariance_edge_branches(self):
        """Test localized covariance edge branches."""
        c2 = make_chain(0.05)
        c2.setRanges({"p0": [-2.0, 2.0], "p1": [-2.0, 2.0]})

        zero_weight_chain = make_chain()
        zero_weight_chain.weights = np.zeros_like(zero_weight_chain.weights)
        cov = gt.get_localized_covariance(zero_weight_chain, c2, param_names=["p0", "p1"])
        self.assertEqual(cov.shape, (2, 2))

        single = make_chain()
        single.samples = single.samples[:1, :]
        single.weights = single.weights[:1]
        cov_single = gt.get_localized_covariance(single, c2, param_names=["p0", "p1"])
        self.assertEqual(cov_single.shape, (2, 2))

        tiny_weight_chain = make_chain()
        tiny_weight_chain.weights = np.full_like(tiny_weight_chain.weights, 1.0e-300)
        cov_mock = gt.get_localized_covariance(tiny_weight_chain, c2, param_names=["p0", "p1"])
        self.assertEqual(cov_mock.shape, (2, 2))

        with patch("tensiometer.gaussian_tension.np.cov", return_value=np.full((2, 2), np.nan)):
            cov_nan = gt.get_localized_covariance(make_chain(), c2, param_names=["p0", "p1"], scale=10.0)
            self.assertTrue(np.all(np.isfinite(cov_nan)))

    def test_localized_covariance_main_path(self):
        """Test localized covariance main path."""
        c1 = make_chain()
        c2 = make_chain(0.05)
        cov = gt.get_localized_covariance(c1, c2, param_names=["p0", "p1"], scale=2.0)
        self.assertEqual(cov.shape, (2, 2))
        mean = np.mean(c2.samples, axis=0)
        offsets = np.array([
            [0.0, 0.0],
            [0.03, -0.01],
            [-0.02, 0.04],
            [0.05, 0.02],
        ])
        near_samples = mean + offsets
        base = np.linspace(0, 5, 50)
        spread = np.column_stack([base, np.sin(base) + 0.1 * base])
        many_samples = np.vstack([spread, near_samples])
        heavy_chain = MCSamples(samples=many_samples, names=["p0", "p1"], labels=["p0", "p1"])
        heavy_chain.setRanges({"p0": [-10, 10], "p1": [-10, 10]})
        with patch("builtins.print") as mock_print:
            gt.get_localized_covariance(heavy_chain, c2, param_names=["p0", "p1"], scale=0.05)
            self.assertTrue(mock_print.called)
        cov2 = gt.get_localized_covariance(c1, c2, param_names=["p0", "p1"], localize_params=["p0"])
        self.assertEqual(cov2.shape[0], 2)

    def test_q_udm_covariance_components_invalid(self):
        """Test Q_UDM covariance components invalid branch."""
        c1 = make_chain()
        c2 = make_chain(0.05)
        with self.assertRaises(ValueError):
            gt.Q_UDM_covariance_components(c1, c2, which="bad")

    def test_q_dm_with_prior_chain(self):
        """Test Q_DM with and without a prior chain."""
        c1 = make_chain()
        c2 = make_chain(0.02)
        prior = make_chain(0.1)
        prior.setRanges({"p0": [-1, 1], "p1": [-2, 2]})
        q, dofs = gt.Q_DM(c1, c2, prior_chain=prior, cutoff=0.01, prior_factor=0.5)
        self.assertTrue(np.isfinite(q))
        self.assertGreaterEqual(dofs, 0)
        q2, dofs2 = gt.Q_DM(c1, c2, prior_chain=None, cutoff=0.01)
        self.assertTrue(np.isfinite(q2))
        self.assertGreaterEqual(dofs2, 0)
        with patch("tensiometer.gaussian_tension.stutils.KL_decomposition",
                   side_effect=[(np.array([2.0, 0.5]), np.eye(2)), (np.array([1.01, 1.02]), np.eye(2))]):
            q3, dofs3 = gt.Q_DM(c1, c2, prior_chain=prior, cutoff=0.01)
            self.assertTrue(np.isfinite(q3))
            self.assertGreaterEqual(dofs3, 0)

    def test_print_cpca_correlation_block(self):
        """Test CPCA correlation block formatting."""
        cpca = {
            "param_names": ["p0", "p1"],
            "param_labels": ["P0", "P1"],
            "conditional_params": ["p0"],
            "marginalized_parameters": [],
            "Neff": 1.0,
            "Neff_spectrum": np.array([1.0, 0.5]),
            "KL_divergence": 0.1,
            "KL_spectrum": np.array([0.1, 0.05]),
            "CPCA_eig": np.array([2.0, 1.1]),
            "CPCA_eigv": np.eye(2),
            "CPCA_projector": np.array([[1.0, 0.2], [0.1, 1.0]]),
            "CPCA_var_contributions": np.ones((2, 2)),
            "CPCA_var_filter": np.ones((2, 2), dtype=bool),
            "normparam": 0,
            "reference_point": np.array([0.0, 0.1]),
            "correlation_mode_parameter": np.array([[0.3, 0.4], [0.5, 0.6]]),
            "correlation_parameter_names": ["p0", "p1"],
        }
        text = gt.print_CPCA_results(cpca, verbose=True, num_modes=2)
        self.assertIn("correlations", text)
        cpca.pop("param_labels")
        cpca["CPCA_var_filter"] = np.array([[True, False], [False, True]])
        text2 = gt.print_CPCA_results(cpca, verbose=False, num_modes=1)
        self.assertIn("CPCA for", text2)
        cpca["CPCA_eig"] = np.array([0.9, 1.0])
        cpca["CPCA_var_contributions"] = np.zeros((2, 2))
        text3 = gt.print_CPCA_results(cpca, verbose=True, num_modes=2)
        self.assertIn("noisy", text3)
        cpca["conditional_params"] = []
        cpca["marginalized_parameters"] = []
        text4 = gt.print_CPCA_results(cpca, verbose=True, num_modes=1)
        self.assertIn("CPCA for", text4)
        log_flags = []

        class Toggle(float):
            """Toggle test suite."""
            counter = 0
            def __new__(cls, value):
                """New."""
                obj = float.__new__(cls, value)
                return obj
            def __sub__(self, other):
                """Sub."""
                return self
            def __gt__(self, other):
                """Gt."""
                Toggle.counter += 1
                res = False if Toggle.counter == 6 else True
                log_flags.append(res)
                return res

        cpca_small = {
            "param_names": ["p0"],
            "param_labels": ["A"],
            "conditional_params": [],
            "marginalized_parameters": [],
            "Neff": 1.0,
            "Neff_spectrum": np.array([1.0]),
            "KL_divergence": 0.1,
            "KL_spectrum": np.array([0.1]),
            "CPCA_eig": np.array([Toggle(2.0)], dtype=object),
            "CPCA_eigv": np.array([[1.0]]),
            "CPCA_projector": np.array([[1.0]]),
            "CPCA_var_contributions": np.array([[1.0]]),
            "CPCA_var_filter": np.array([[True]]),
            "normparam": None,
        }
        text5 = gt.print_CPCA_results(cpca_small, verbose=True, num_modes=1)
        self.assertIn("parameter combinations", text5)
        self.assertIn(False, log_flags)
        self.assertTrue(log_flags[-1] is False)

    def test_q_udm_get_cutoff_success(self):
        """Test Q_UDM cutoff bracketing success and failure."""
        dummy_names = ["p0", "p1"]
        with patch("tensiometer.gaussian_tension._check_param_names", return_value=dummy_names), \
                patch("tensiometer.gaussian_tension.Q_UDM_KL_components",
                      return_value=(np.array([2.0, 3.0]), np.eye(2), dummy_names)), \
                patch("tensiometer.gaussian_tension.get_Neff", side_effect=[1.0, 1.0, 1.0]):
            cutoff, eig, eigv, names = gt.Q_UDM_get_cutoff(make_chain(), make_chain(), make_chain())
        self.assertGreater(cutoff, 0)
        self.assertEqual(list(names), dummy_names)
        # failure branch when cutoff cannot be bracketed
        with patch("tensiometer.gaussian_tension.Q_UDM_KL_components",
                   return_value=(np.array([1.1, 1.1]), np.eye(2), dummy_names)), \
                patch("tensiometer.gaussian_tension.get_Neff", side_effect=[5.0, 5.0, 5.0]):
            with self.assertRaises(ValueError):
                gt.Q_UDM_get_cutoff(make_chain(), make_chain(), make_chain())
        c1 = MCSamples(samples=np.array([[0.1]]), names=["a"], labels=["a"])
        c2 = MCSamples(samples=np.array([[0.2]]), names=["b"], labels=["b"])
        with self.assertRaises(ValueError):
            gt.Q_UDM_get_cutoff(c1, c2, c2)

    def test_q_udm_fisher_covariance_branches(self):
        """Test Q_UDM fisher and covariance branches."""
        dummy_names = ["p0", "p1"]
        eig = np.array([2.0, 3.0])
        eigv = np.array([[1.0, 0.0], [0.0, 1.0]])
        with patch("tensiometer.gaussian_tension.Q_UDM_KL_components", return_value=(eig, eigv, dummy_names)):
            names, eig_out, frac_fisher, fisher = gt.Q_UDM_fisher_components(make_chain(), make_chain(), which="2")
            self.assertEqual(names, dummy_names)
            self.assertEqual(eig_out.tolist(), eig.tolist())
            names, eig_out, frac_fisher, fisher = gt.Q_UDM_fisher_components(make_chain(), make_chain(), which="12")
            self.assertEqual(frac_fisher.shape[0], 2)
            names, eig_out, frac_cov = gt.Q_UDM_covariance_components(make_chain(), make_chain(), which="12")
            self.assertEqual(frac_cov.shape[0], 2)
            names, eig_out, frac_fisher, fisher = gt.Q_UDM_fisher_components(make_chain(), make_chain(), which="1")
            self.assertTrue(np.all(np.isfinite(frac_fisher)))
            names, eig_out, frac_cov = gt.Q_UDM_covariance_components(make_chain(), make_chain(), which="2")
            self.assertTrue(np.all(np.isfinite(frac_cov)))
            names, eig_out, frac_cov = gt.Q_UDM_covariance_components(make_chain(), make_chain(), which="1")
            self.assertTrue(np.all(np.isfinite(frac_cov)))

    def test_q_udm_components_errors(self):
        """Test Q_UDM component errors for missing inputs."""
        c1 = MCSamples(samples=np.array([[0.1]]), names=["a"], labels=["a"])
        c2 = MCSamples(samples=np.array([[0.2]]), names=["b"], labels=["b"])
        with self.assertRaises(ValueError):
            gt.Q_UDM_KL_components(c1, c2)
        with self.assertRaises(ValueError):
            gt.Q_UDM_fisher_components(c1, c2)
        with self.assertRaises(ValueError):
            gt.Q_UDM_covariance_components(c1, c2)
        with patch("tensiometer.gaussian_tension.Q_UDM_KL_components",
                   return_value=(np.array([1.5, 0.5]), np.eye(2), ["p0", "p1"])):
            q, dofs = gt.Q_UDM(make_chain(), make_chain())
            self.assertTrue(np.isfinite(q))
            self.assertGreaterEqual(dofs, 1)

    def test_get_map_loglike_bestfit_empty_chi(self):
        """Exercise MAP loglike fallbacks when best-fit data are missing."""

        class EmptyChiBestFit:
            """Best-fit mock without chi-squared entries."""

            def __init__(self):
                """Initialize best-fit values."""
                self.chiSquareds = []
                self.logLike = 5.0

            def getParamDict(self):
                """Return prior contribution."""
                return {"prior": 1.0}

        class ChainWithPriorBestFit:
            """Chain exposing explicit best-fit data with a prior term."""

            def getBestFit(self, max_posterior=True):
                """Return stored best-fit state."""
                return EmptyChiBestFit()

        self.assertEqual(gt.get_MAP_loglike(ChainWithPriorBestFit(), feedback=False), -2.0)

        class EmptyStats:
            """Stats container with no chi-squared entries."""

            def list(self):
                """Return an empty chi-squared list."""
                return []

            @property
            def logLike_sample(self):
                """Provide the sampled log-likelihood."""
                return -2.5

        class ChainWithStatsFallback:
            """Chain that triggers the stats fallback path."""

            def getBestFit(self, max_posterior=True):
                """Raise to force the fallback path."""
                raise RuntimeError("fail")

            def getLikeStats(self):
                """Return stats with no chi-squared labels."""
                return EmptyStats()

        self.assertEqual(gt.get_MAP_loglike(ChainWithStatsFallback(), feedback=False), 1.25)

        class ChiObj:
            """Container for a chi-squared value."""

            def __init__(self, value):
                """Store the chi-squared value."""
                self.chisq = value

        class ChiBestFit:
            """Best-fit mock with chi-squared entries."""

            def __init__(self):
                """Create chi-squared entries."""
                self.chiSquareds = [("d", ChiObj(4.0)), ("d2", ChiObj(2.0))]
                self.logLike = 0.0

            def getParamDict(self):
                """Return an empty parameter mapping."""
                return {}

        class ChainWithChiBestFit:
            """Chain returning best-fit info with chi-squared values."""

            def getBestFit(self, max_posterior=True):
                """Return chi-squared best-fit entries."""
                return ChiBestFit()

        self.assertEqual(gt.get_MAP_loglike(ChainWithChiBestFit(), feedback=False), -3.0)
        self.assertEqual(ChiBestFit().getParamDict(), {})

        class StatsWithPrior:
            """Stats including a prior chi-squared term."""

            def list(self):
                """Return chi-squared labels with a prior entry."""
                return ["chi2_data", "chi2_prior"]

            def parWithName(self, name):
                """Return parameter info for a chi-squared entry."""

                class Param:
                    """Parameter wrapper holding a best-fit sample."""

                    def __init__(self, value):
                        """Set the best-fit sample value."""
                        self.bestfit_sample = value

                return Param(1.0)

        class ChainWithPriorStats:
            """Chain raising on best fit and falling back to stats with prior."""

            def getBestFit(self, max_posterior=True):
                """Raise to exercise fallback."""
                raise RuntimeError("fail")

            def getLikeStats(self):
                """Return stats with prior chi-squared entries."""
                return StatsWithPrior()

        self.assertEqual(gt.get_MAP_loglike(ChainWithPriorStats(), feedback=False), -0.5)

        class NoStatsChain:
            """Chain raising on best fit and returning no stats."""

            def getBestFit(self, max_posterior=True):
                """Raise to trigger missing stats fallback."""
                raise RuntimeError("fail")

            def getLikeStats(self):
                """Return no stats information."""
                return None

        self.assertEqual(gt.get_MAP_loglike(NoStatsChain(), feedback=False), 0.0)

        class StatsWithoutPrior:
            """Stats object with multiple chi-squared entries and no prior."""

            def list(self):
                """Return chi-squared labels without a prior entry."""
                return ["chi2_data", "chi2_extra"]

            def parWithName(self, name):
                """Return parameter info for a chi-squared entry."""

                class Param:
                    """Parameter wrapper holding a best-fit sample."""

                    def __init__(self, value):
                        """Set the best-fit sample value."""
                        self.bestfit_sample = value

                return Param(2.0)

        class ChainWithStatsOnly:
            """Chain raising on best fit and using stats without prior."""

            def getBestFit(self, max_posterior=True):
                """Raise to force stats-only path."""
                raise RuntimeError("fail")

            def getLikeStats(self):
                """Return stats without prior information."""
                return StatsWithoutPrior()

        self.assertEqual(gt.get_MAP_loglike(ChainWithStatsOnly(), feedback=False), -2.0)

        class ChainPrintingWarning:
            """Chain exercising feedback path without stats."""

            def getBestFit(self, max_posterior=True):
                """Raise to emit warning when feedback is enabled."""
                raise RuntimeError("fail")

            def getLikeStats(self):
                """Return no stats information."""
                return None

        self.assertEqual(gt.get_MAP_loglike(ChainPrintingWarning(), feedback=True), 0.0)

        class BestFitWithoutPrior:
            """Best-fit mock without chi-squared entries or priors."""

            def __init__(self):
                """Initialize best-fit values."""
                self.chiSquareds = []
                self.logLike = 3.0

            def getParamDict(self):
                """Return an empty parameter dictionary."""
                return {}

        class ChainWithoutPrior:
            """Chain returning a best fit with no prior contributions."""

            def getBestFit(self, max_posterior=True):
                """Return best-fit values without priors."""
                return BestFitWithoutPrior()

        self.assertEqual(gt.get_MAP_loglike(ChainWithoutPrior(), feedback=False), -1.5)

    def test_get_neff_localize_and_linear_cpca_chains(self):
        """Test Neff localization and linear CPCA chain helper."""
        chain = make_chain()
        prior = make_chain()
        prior.setRanges({"p0": [-1, 1], "p1": [-2, 2]})
        neff = gt.get_Neff(chain, prior_chain=prior, localize=True, param_names=["p0", "p1"])
        self.assertTrue(np.isfinite(neff))
        res = gt.linear_CPCA_chains(chain, chain, param_names=["p0", "p1"], dimensional_reduce=False)
        self.assertIn("CPCA_eig", res)
        fisher = np.eye(2)
        res_lin = gt.linear_CPCA(fisher, fisher, ["p0", "p1"], dimensional_reduce=False)
        self.assertIn("CPCA_projector", res_lin)

    def test_linear_cpca_validation_errors(self):
        """Test linear CPCA validation errors."""
        fisher = np.eye(2)
        with self.assertRaises(ValueError):
            gt.linear_CPCA(fisher, np.eye(3), ["p0", "p1"])
        with self.assertRaises(ValueError):
            gt.linear_CPCA(fisher, fisher, ["p0", "p1"], conditional_params=["bad"])
        with self.assertRaises(ValueError):
            gt.linear_CPCA(fisher, fisher, ["p0", "p1"], marginalized_parameters=["bad"])
        res = gt.linear_CPCA(fisher, fisher, ["p0", "p1"],
                             conditional_params=["p1"], marginalized_parameters=[])
        self.assertIn("CPCA_eig", res)

    def test_print_cpca_verbose_blocks(self):
        """Test CPCA verbose print blocks."""
        cpca = {
            "param_names": ["p0", "p1"],
            "param_labels": ["A", "B"],
            "conditional_params": ["p1"],
            "marginalized_parameters": ["p0"],
            "Neff": 1.0,
            "Neff_spectrum": np.array([0.5, 0.5]),
            "KL_divergence": 0.1,
            "KL_spectrum": np.array([0.05, 0.05]),
            "CPCA_eig": np.array([2.0, 0.8]),
            "CPCA_eigv": np.eye(2),
            "CPCA_projector": np.eye(2),
            "CPCA_var_contributions": np.ones((2, 2)),
            "CPCA_var_filter": np.ones((2, 2), dtype=bool),
            "normparam": None,
        }
        text = gt.print_CPCA_results(cpca, verbose=True, num_modes=2)
        self.assertIn("fixed parameters", text)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
