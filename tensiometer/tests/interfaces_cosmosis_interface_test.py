"""Tests for Cosmosis interface helpers."""

#########################################################################################################
# Imports

import os
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from getdist.mcsamples import MCSamples

import tensiometer.interfaces.cosmosis_interface as ci

#########################################################################################################
# Helper functions


def write_chain(path, lines, data):
    """Write a Cosmosis-like chain file.

    :param path: filesystem path to write.
    :param lines: header lines to prepend.
    :param data: rows of numerical samples.
    :returns: ``None``.
    """
    with open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")
        for row in data:
            f.write(" ".join(str(x) for x in row) + "\n")


#########################################################################################################
# Interface tests


class TestCosmosisInterface(unittest.TestCase):

    """Cosmosis interface test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.chain_dir = self.here + "/../../test_chains/"

    def test_mcsamples_from_cosmosis(self):
        """Test Cosmosis chain import helper."""
        chain_name = self.chain_dir + "DES_multinest_cosmosis"
        chain = ci.MCSamplesFromCosmosis(chain_name)
        self.assertIsNotNone(chain)

    def test_get_cosmosis_info_and_params(self):
        """Test cosmosis info parsing and parameter helpers."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("# a b c\n# sampler = emcee\n# run_name = test_run\n")
            tmp_path = tmp.name
        info = ci.get_cosmosis_info(tmp_path)
        self.assertIn("sampler = emcee", info)
        names = ci.get_param_names(info)
        self.assertEqual(names, ["a", "b", "c"])
        labels = ci.get_param_labels(info, names, {"a": "A"})
        self.assertEqual(labels[0], "A")
        labels_none = ci.get_param_labels(info, names, None)
        self.assertIsNone(labels_none)
        sampler_type, sampler = ci.get_sampler_type(info)
        self.assertEqual(sampler_type, "mcmc")
        tag = ci.get_name_tag(info)
        self.assertEqual(tag, "test_run")
        os.remove(tmp_path)

    def test_get_ranges(self):
        """Test parameter range parsing."""
        info = ["param1 param2", "[sec]", "param1 = 0.1 0.0 1.0", "param2 = -1 0.0 2"]
        ranges = ci.get_ranges(info, ["sec--param1", "sec--param2", "other--x"])
        self.assertIn("sec--param1", ranges)
        self.assertEqual(ranges["sec--param1"], [0.1, 1.0])
        self.assertNotIn("other--x", ranges)

    def test_sampler_type_invalid(self):
        """Test invalid sampler type handling."""
        info = ["# sampler = unknown"]
        with self.assertRaises(ValueError):
            ci.get_sampler_type(ci.get_cosmosis_info(self._make_temp_file(info)))
        with self.assertRaises(ValueError):
            ci.MCSamplesFromCosmosis("nonexistent_path")

    def _make_temp_file(self, lines):
        """Create a temporary file with the provided lines.

        :param lines: iterable of lines to write.
        :returns: path to the created file.
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            for ln in lines:
                tmp.write(f"{ln}\n")
            return tmp.name

    def test_mcsamples_from_cosmosis_nested(self):
        """Test nested Cosmosis chain path handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "nested_chain.txt")
            lines = [
                "# weight a post",
                "# nsample=2",
                "# sampler = multinest",
                "# run_name = nested_run",
            ]
            data = [
                [0.2, 1.0, -1.0],
                [0.8, 2.0, -2.0],
            ]
            write_chain(chain_path, lines, data)
            chain = ci.MCSamplesFromCosmosis(chain_path[:-4])
            self.assertEqual(chain.name_tag, "nested_run")
            self.assertEqual(chain.samples.shape[0], 2)
            dir_with_chain = os.path.join(tmpdir, "dirchain")
            os.mkdir(dir_with_chain)
            os.replace(chain_path, os.path.join(dir_with_chain, "chain.txt"))
            chain_dir = ci.MCSamplesFromCosmosis(dir_with_chain)
            self.assertEqual(chain_dir.samples.shape[0], 2)
            folder = os.path.join(tmpdir, "folder")
            os.mkdir(folder)
            with self.assertRaises(ValueError):
                ci.MCSamplesFromCosmosis(folder)

    def test_mcsamples_from_cosmosis_mcmc_importance(self):
        """Test MCMC and importance-sampler Cosmosis paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "importance_chain.txt")
            lines = [
                "# old_weight log_weight a post",
                "# sampler = importance",
                "# run_name = imp_run",
            ]
            data = [
                [1.0, np.log(2.0), 0.5, -1.0],
                [0.0, np.log(1.0), 0.7, -2.0],
            ]
            write_chain(chain_path, lines, data)
            chain = ci.MCSamplesFromCosmosis(chain_path[:-4])
            self.assertTrue(np.all(chain.weights > 0))
            chain_path2 = os.path.join(tmpdir, "importance_chain2.txt")
            lines2 = [
                "# old_log_weight log_weight a post",
                "# sampler = importance",
                "# run_name = imp_run2",
            ]
            data2 = [
                [np.log(1.5), np.log(2.0), 0.5, -1.0],
                [np.log(1.1), np.log(1.5), 0.7, -2.0],
            ]
            write_chain(chain_path2, lines2, data2)
            chain2 = ci.MCSamplesFromCosmosis(chain_path2[:-4])
            self.assertTrue(np.all(chain2.weights >= 0))
            bad_path = os.path.join(tmpdir, "bad_mcmc.txt")
            write_chain(bad_path, ["# weight a post", "# sampler = emcee"], [[1.0, 0.1, -1.0]])
            with self.assertRaises(ValueError):
                ci.MCSamplesFromCosmosis(bad_path[:-4])

    def test_mcsamples_from_cosmosis_uncorrelated(self):
        """Test uncorrelated Cosmosis chain imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "uncorr_chain.txt")
            lines = [
                "# a post",
                "# sampler = apriori",
                "# run_name = uncorr_run",
            ]
            data = [
                [0.1, -1.0],
                [0.2, -2.0],
            ]
            write_chain(chain_path, lines, data)
            chain = ci.MCSamplesFromCosmosis(chain_path[:-4], param_label_dict={"a": "A"})
            self.assertTrue(np.all(chain.weights == 1.0))

    def test_polish_samples(self):
        """Test NaN filtering in polish_samples."""
        samples = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        chain = MCSamples(samples=samples, names=["x", "y"])
        cleaned = ci.polish_samples(chain)
        self.assertEqual(cleaned.samples.shape[1], 1)
        self.assertTrue(np.all(np.isfinite(cleaned.samples)))

    def test_get_maximum_likelihood(self):
        """Test maximum-likelihood extraction from chain files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            min_path = os.path.join(tmpdir, "chain_pmaxlike.txt")
            lines = [
                "# param1 param2 post prior like",
                "# sampler = max_like",
            ]
            data = [[0.1, 0.2, -5.0, -1.0, -4.0]]
            write_chain(min_path, lines, data)
            bf = ci.get_maximum_likelihood(None, True, min_path[:-4], {"param1": "p1"}, {"param1": "P1"})
            self.assertEqual(bf.logLike, 5.0)
            self.assertEqual(bf.names[0].name, "p1")
            min_path2 = os.path.join(tmpdir, "chain_pmaxlike2.txt")
            write_chain(min_path2, ["# weigth post", "# sampler = max_like"], [[2.0, -3.0]])
            bf2 = ci.get_maximum_likelihood(None, True, min_path2[:-4], None, None)
            self.assertEqual(bf2.weight, 2.0)
            wrong_path = os.path.join(tmpdir, "chain_wrong.txt")
            write_chain(wrong_path, ["# param1 post", "# sampler = unknown"], [[0.1, -1.0]])
            with self.assertRaises(ValueError):
                ci.get_maximum_likelihood(None, True, wrong_path[:-4], None, None)
            empty_dir = os.path.join(tmpdir, "empty")
            os.mkdir(empty_dir)
            with self.assertRaises(ValueError):
                ci.get_maximum_likelihood(None, True, empty_dir, None, None)
            min_path3 = os.path.join(tmpdir, "chain_pmaxlike3.txt")
            write_chain(min_path3, ["# param1", "# sampler = max_like"], [[0.1]])
            with self.assertRaises(ValueError):
                ci.get_maximum_likelihood(None, True, min_path3[:-4], None, None)

    def test_nested_missing_weight_error(self):
        """Test nested-sampler missing weight error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "nested_bad.txt")
            lines = [
                "# a post",
                "# nsample=1",
                "# sampler = polychord",
            ]
            data = [[0.1, -1.0]]
            write_chain(chain_path, lines, data)
            with self.assertRaises(ValueError):
                ci.MCSamplesFromCosmosis(chain_path[:-4])

    def test_log_weight_nested_and_no_sampler(self):
        """Test log_weight nested branch and missing sampler info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "nested_log.txt")
            lines = [
                "# log_weight a post",
                "# nsample=1",
                "# sampler = multinest",
                "# run_name = log_nested",
            ]
            data = [[np.log(1.0), 0.2, -0.5]]
            write_chain(chain_path, lines, data)
            chain = ci.MCSamplesFromCosmosis(
                chain_path[:-4],
                param_label_dict={"log_weight": "log_weight", "post": "post", "a": "a"},
            )
            self.assertTrue(np.all(np.isfinite(chain.weights)))
            with self.assertRaises(Exception):
                ci.get_sampler_type(["param1 param2"])

    def test_mcsamples_from_cosmosis_importance_missing_old_weights(self):
        """Test importance-sampler missing old weights handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "importance_missing.txt")
            write_chain(
                chain_path,
                ["# log_weight a post", "# sampler = importance"],
                [[0.1, 1.0, -1.0]],
            )
            with self.assertRaises(Exception):
                ci.MCSamplesFromCosmosis(chain_path[:-4])

    def test_mcsamples_from_cosmosis_mcmc_custom_name_and_mapping(self):
        """Test MCMC name tag override and parameter remapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "emcee_chain.txt")
            lines = [
                "# x post",
                "# sampler = emcee",
                "# run_name = default_run",
            ]
            data = [
                [0.3, -1.0],
                [0.3, -1.0],
                [0.4, -2.0],
            ]
            write_chain(chain_path, lines, data)
            chain = ci.MCSamplesFromCosmosis(
                chain_path[:-4],
                param_label_dict={"x": "X"},
                param_name_dict={"x": "alpha"},
                name_tag="custom_name",
            )
            self.assertEqual(chain.name_tag, "custom_name")
            self.assertIn("alpha", chain.getParamNames().list())
            self.assertTrue(np.any(chain.weights > 1.0))

    def test_mcsamples_from_cosmosis_unknown_sampler(self):
        """Test unknown sampler error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "unknown_chain.txt")
            lines = [
                "# y post",
            ]
            data = [[0.1, -1.0]]
            write_chain(chain_path, lines, data)
            with self.assertRaises(Exception):
                ci.MCSamplesFromCosmosis(chain_path[:-4])

    def test_polish_samples_with_ranges(self):
        """Test polish_samples with range metadata."""
        class DummyRanges:
            """Dummy Ranges test suite."""
            def __init__(self):
                """Init."""
                self.fixed = []

            def setFixed(self, name, value):
                """Set Fixed."""
                self.fixed.append((name, value))

        class DummyParamNames:
            """Dummy Param Names test suite."""
            def __init__(self, names):
                """Init."""
                self.names = [types.SimpleNamespace(name=n) for n in names]

            def deleteIndices(self, fixed):
                """Delete Indices."""
                for ix in sorted(fixed, reverse=True):
                    del self.names[ix]

            def list(self):
                """List."""
                return [n.name for n in self.names]

            def parsWithNames(self, names):
                """Pars With Names."""
                return self.names

        dummy_chain = types.SimpleNamespace()
        dummy_chain.samples = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        dummy_chain.ranges = DummyRanges()
        dummy_chain.paramNames = DummyParamNames(["x", "y"])
        dummy_chain.deleteFixedParams = lambda: None
        dummy_chain.changeSamples = lambda arr: setattr(dummy_chain, "samples", arr)
        dummy_chain._getParamIndices = lambda: None
        dummy_chain.filter = lambda where: setattr(dummy_chain, "samples", dummy_chain.samples[where])
        dummy_chain.getParamNames = lambda: dummy_chain.paramNames
        cleaned = ci.polish_samples(dummy_chain)
        self.assertEqual(cleaned.samples.shape[1], 1)
        self.assertEqual(dummy_chain.ranges.fixed[0][0], "x")
        self.assertEqual(dummy_chain.paramNames.list(), ["y"])
        self.assertEqual(len(dummy_chain.paramNames.parsWithNames(["x"])), 1)

    def test_polish_samples_without_ranges(self):
        """Test polish_samples when ranges are missing."""
        dummy_chain = types.SimpleNamespace()
        dummy_chain.samples = np.array([[np.nan], [np.nan]])
        dummy_chain.paramNames = type("PN", (), {})()
        dummy_chain.paramNames.names = [types.SimpleNamespace(name="x")]
        dummy_chain.paramNames.deleteIndices = lambda fixed: None
        dummy_chain.paramNames.list = lambda: ["x"]
        dummy_chain.paramNames.parsWithNames = lambda names: dummy_chain.paramNames.names
        dummy_chain.deleteFixedParams = lambda: None
        dummy_chain.changeSamples = lambda arr: setattr(dummy_chain, "samples", arr)
        dummy_chain._getParamIndices = lambda: None
        dummy_chain.filter = lambda where: setattr(dummy_chain, "samples", dummy_chain.samples[where])
        dummy_chain.getParamNames = lambda: dummy_chain.paramNames
        cleaned = ci.polish_samples(dummy_chain)
        self.assertEqual(cleaned.samples.shape[1], 0)

    def test_get_maximum_likelihood_invalid_inputs(self):
        """Test invalid-input handling in get_maximum_likelihood."""
        with self.assertRaises(ValueError):
            ci.get_maximum_likelihood(None, True, "missing_path", None, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            wrong_sampler = os.path.join(tmpdir, "wrong_min.txt")
            write_chain(wrong_sampler, ["# param post", "# sampler = emcee"], [[0.1, -1.0]])
            with self.assertRaises(ValueError):
                ci.get_maximum_likelihood(None, True, wrong_sampler[:-4], None, None)

    def test_mcsamples_from_cosmosis_with_bestfit_override(self):
        """Test best-fit override from max-like chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "main_chain.txt")
            write_chain(chain_path, ["# z post", "# sampler = emcee"], [[1.0, -1.0], [1.2, -1.1]])
            min_path = os.path.join(tmpdir, "chain_pmaxlike.txt")
            write_chain(min_path, ["# z post prior like", "# sampler = max_like"], [[1.5, -2.0, -0.5, -1.5]])
            chain = ci.MCSamplesFromCosmosis(
                chain_path[:-4],
                chain_min_root=tmpdir,
                param_name_dict={"z": "zed"},
                param_label_dict={"z": "Z"},
            )
            best = chain.getBestFit(max_posterior=True)
            self.assertAlmostEqual(best.logLike, 2.0)
            self.assertEqual(best.names[0].name, "zed")

    def test_mcsamples_from_cosmosis_range_mapping(self):
        """Test range mapping during Cosmosis chain import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "range_chain.txt")
            lines = [
                "# sec--p1 post",
                "# [sec]",
                "# p1 = 0.0 0.0 1.0",
                "# sampler = emcee",
            ]
            data = [[0.2, -1.0]]
            write_chain(chain_path, lines, data)
            chain = ci.MCSamplesFromCosmosis(chain_path[:-4], param_name_dict={"sec--p1": "p1"})
            self.assertIn("p1", chain.ranges.names)

    def test_mcsamples_from_cosmosis_param_mapping_no_match(self):
        """Test parameter mapping fallback when no match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "nomap_chain.txt")
            write_chain(chain_path, ["# q post", "# sampler = emcee"], [[0.1, -1.0], [0.2, -1.1]])
            chain = ci.MCSamplesFromCosmosis(chain_path[:-4], param_name_dict={"unused": "x"})
            self.assertIn("q", chain.getParamNames().list())

    def test_mcsamples_from_cosmosis_uncorrelated_no_labels(self):
        """Test uncorrelated chain import without labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "uncorr_chain2.txt")
            lines = [
                "# b post",
                "# sampler = apriori",
            ]
            data = [[0.3, -0.5], [0.4, -0.6]]
            write_chain(chain_path, lines, data)
            chain = ci.MCSamplesFromCosmosis(chain_path[:-4])
            label = chain.getParamNames().parsWithNames(["b"])[0].label
            self.assertIn(label, ("", "b", None))

    def test_mcsamples_from_cosmosis_unknown_sampler_branch(self):
        """Test unknown sampler branch during import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "unknown_branch.txt")
            write_chain(chain_path, ["# c post", "# sampler = emcee"], [[0.1, -0.2]])
            with patch("tensiometer.interfaces.cosmosis_interface.get_sampler_type", return_value=("odd", "odd")):
                with self.assertRaises(ValueError):
                    ci.MCSamplesFromCosmosis(chain_path[:-4])

    def test_get_maximum_likelihood_without_prior_like(self):
        """Test maximum likelihood when prior/like fields are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            min_path = os.path.join(tmpdir, "min_simple.txt")
            write_chain(min_path, ["# c post", "# sampler = max_like"], [[0.2, -0.5]])
            best = ci.get_maximum_likelihood(None, True, min_path[:-4], None, None)
            self.assertAlmostEqual(best.logLike, 0.5)
            self.assertEqual(best.weight, 1.0)

    def test_get_maximum_likelihood_post_branch_false(self):
        """Test post-branch fallback when post is absent."""
        class FlakyList(list):
            """Flaky List test suite."""
            def __contains__(self, item):
                """Contains."""
                if item == "post":
                    self.flag = getattr(self, "flag", False)
                    if not self.flag:
                        self.flag = True
                        return True
                    return False
                return super().__contains__(item)

        with tempfile.TemporaryDirectory() as tmpdir:
            min_path = os.path.join(tmpdir, "min_branch.txt")
            write_chain(min_path, ["# dummy"], [[1.0]])
            fake_names = FlakyList(["post"])
            with patch("tensiometer.interfaces.cosmosis_interface.get_cosmosis_info", return_value=["# dummy"]):
                with patch("tensiometer.interfaces.cosmosis_interface.get_param_names", return_value=fake_names):
                    with patch("tensiometer.interfaces.cosmosis_interface.get_param_labels", return_value=None):
                        with patch("tensiometer.interfaces.cosmosis_interface.get_sampler_type", return_value=("max_like", "max_like")):
                            with patch("tensiometer.interfaces.cosmosis_interface.get_ranges", return_value={}):
                                with patch("tensiometer.interfaces.cosmosis_interface.loadNumpyTxt", return_value=np.array([[1.0]])):
                                    best = ci.get_maximum_likelihood(None, True, min_path[:-4], None, None)
        self.assertEqual(best.logLike, -1.0)
#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
