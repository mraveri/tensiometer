"""Tests for statistical utility helpers."""

#########################################################################################################
# Imports

import unittest

import numpy as np
from scipy.stats import multivariate_normal

from getdist import MCSamples

import tensiometer.utilities.stats_utilities as stut

#########################################################################################################
# Helper functions

def helper_mcsamples(ndim=4, nsamp=10000):
    """Build an MCSamples instance for statistics tests.

    :param ndim: number of parameters.
    :param nsamp: number of samples to generate.
    :returns: MCSamples with Gaussian draws.
    """
    random_state = np.random.default_rng(0)
    a_mat = random_state.random((ndim, ndim))
    cov = np.dot(a_mat, a_mat.T)

    distribution = multivariate_normal([0] * ndim, cov)
    samps = distribution.rvs(nsamp)
    loglikes = -distribution.logpdf(samps)

    names = [f"x{i}" for i in range(ndim)]
    labels = [f"x_{i}" for i in range(ndim)]

    samples = MCSamples(samples=samps, names=names, labels=labels, loglikes=loglikes)
    return samples

#########################################################################################################
# Confidence-to-sigma tests


class TestConfidenceToSigma(unittest.TestCase):

    """Confidence-to-sigma test suite."""
    def setUp(self):
        """Set up test fixtures."""

    def test_from_confidence_to_sigma_result(self):
        """Test confidence-to-sigma conversion."""
        result = stut.from_confidence_to_sigma(np.array([0.68, 0.95, 0.997]))
        known_result = np.array([0.99445788, 1.95996398, 2.96773793])
        assert np.allclose(result, known_result)

    def test_from_sigma_to_confidence_result(self):
        """Test sigma-to-confidence conversion."""
        result = stut.from_sigma_to_confidence(np.array([1., 2., 3.]))
        known_result = np.array([0.68268949, 0.95449974, 0.9973002])
        assert np.allclose(result, known_result)

    def test_sigma_confidence_inverse(self):
        """Test sigma and confidence inversion."""
        test_numbers = np.arange(1, 6)
        test_confidence = stut.from_sigma_to_confidence(test_numbers)
        test_sigma = stut.from_confidence_to_sigma(test_confidence)
        assert np.allclose(test_numbers, test_sigma)

    def test_errors(self):
        """Test invalid input errors."""
        with self.assertRaises(ValueError):
            stut.from_confidence_to_sigma(-1.)
        with self.assertRaises(ValueError):
            stut.from_confidence_to_sigma(2.)
        with self.assertRaises(ValueError):
            stut.from_sigma_to_confidence(-1.)

#########################################################################################################
# Chi-squared to sigma tests


class TestChi2ToSigma(unittest.TestCase):

    """Chi-squared to sigma test suite."""
    def setUp(self):
        """Set up test fixtures."""

    def test_values(self):
        """Test chi-squared conversions."""
        assert np.allclose(stut.from_chi2_to_sigma(1., 1.), 1.0)
        assert np.allclose(stut.from_chi2_to_sigma(5.0, 2.),
                           stut.from_chi2_to_sigma(5.0, 2., 1000))
        assert np.allclose(stut.from_chi2_to_sigma(20.0, 2.),
                           stut.from_chi2_to_sigma(20.0, 2., 9))

    def test_errors(self):
        """Test chi-squared input errors."""
        with self.assertRaises(ValueError):
            stut.from_chi2_to_sigma(-2., 2.)
        with self.assertRaises(ValueError):
            stut.from_chi2_to_sigma(2., -2.)

#########################################################################################################
# KL decomposition tests


class TestKlDecomposition(unittest.TestCase):

    """KL decomposition test suite."""
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(0)
        self.mat_1 = np.random.rand(10)
        self.mat_2 = np.random.rand(10)
        self.mat_1 = stut.vector_to_PDM(self.mat_1)
        self.mat_2 = stut.vector_to_PDM(self.mat_2)

    def test_values(self):
        """Test KL decomposition values."""
        stut.KL_decomposition(self.mat_1, self.mat_2)
        kl_eig, kl_eigv = stut.KL_decomposition(self.mat_1, np.identity(self.mat_2.shape[0]))
        eig, eigv = np.linalg.eigh(self.mat_1)
        assert np.allclose(eig, kl_eig)
        assert np.allclose(eigv, kl_eigv)

    def test_errors(self):
        """Test KL decomposition input errors."""
        d = 10
        wrong_mat = np.random.rand(d, d)
        right_mat = stut.vector_to_PDM(np.random.rand(d*(d+1)//2))
        with self.assertRaises(ValueError):
            stut.KL_decomposition(right_mat, wrong_mat)


#########################################################################################################
# QR inverse tests


class TestQrInverse(unittest.TestCase):

    """QR inverse test suite."""
    def setUp(self):
        """Set up test fixtures."""
        d = 10
        self.mat = stut.vector_to_PDM(np.random.rand(d*(d+1)//2))

    def test_values(self):
        """Test QR inverse values."""
        assert np.allclose(np.linalg.inv(self.mat), stut.QR_inverse(self.mat))

#########################################################################################################


class TestClopperPearsonBinomialTrial(unittest.TestCase):

    """Clopper-Pearson binomial trial test suite."""
    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_values(self):
        """Test values."""
        low, high = stut.clopper_pearson_binomial_trial(1., 2.)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, high)

    def test_errors(self):
        """Test error handling."""
        pass

#########################################################################################################


class TestPdmVectorization(unittest.TestCase):

    """Positive-definite matrix vectorization test suite."""
    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_values(self):
        # generate a random vector between -1, 1 (seeded so reproducible):
        """Test values."""
        np.random.seed(0)
        # sweep dimensions from low to medium
        for d in range(2, 20):
            num = d*(d+1)//2
            # get some random matrices:
            for i in range(10):
                vec = 2.*np.random.rand(num) -1.
                # get the corresponding PDM matrix:
                mat = stut.vector_to_PDM(vec)
                # check that it is positive definite:
                assert np.all(np.linalg.eig(mat)[0] > 0)
                # transform back. This can be different from the previous one
                # because of many discrete symmetries in defining the
                # eigenvectors
                vec2 = stut.PDM_to_vector(mat)
                # transform again. This has to be equal, discrete symmetries for
                # eigenvectors do not matter once they are paired with eigenvalues:
                mat2 = stut.vector_to_PDM(vec2)
                assert np.allclose(mat, mat2)

    def test_errors(self):
        """Test error handling."""
        pass

#########################################################################################################


class TestMakeList(unittest.TestCase):

    """Make-list helper test suite."""
    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_values(self):
        """Test values."""
        _input = ['1', '2']
        _list = stut.make_list(_input)
        self.assertEqual(_list, _input)
        _list = stut.make_list('1')
        self.assertEqual(_list, ['1'])

    def test_errors(self):
        """Test error handling."""
        self.assertEqual(stut.make_list(('a', 'b')), ('a', 'b'))

#########################################################################################################


class TestBernoulliThin(unittest.TestCase):

    """Bernoulli thinning test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.chain = helper_mcsamples()

    def test_values(self):
        """Test values."""
        thinned = stut.bernoulli_thin(self.chain)
        self.assertLessEqual(len(thinned.weights), len(self.chain.weights))
        self.assertTrue(np.all(thinned.weights >= 0))

    def test_errors(self):
        """Test error handling."""
        pass

#########################################################################################################


class TestRandomSamplesReshuffle(unittest.TestCase):

    """Random-samples reshuffle test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.chain = helper_mcsamples()

    def test_values(self):
        """Test values."""
        before = self.chain.weights.copy()
        stut.random_samples_reshuffle(self.chain)
        self.chain.loglikes = None
        reshuffled = stut.random_samples_reshuffle(self.chain)
        self.assertEqual(len(before), len(reshuffled.weights))
        # reshuffle should produce a permutation of weights
        self.assertTrue(np.array_equal(np.sort(before), np.sort(reshuffled.weights)))

    def test_errors(self):
        """Test error handling."""
        pass

#########################################################################################################


class TestWhitenSamples(unittest.TestCase):

    """Whiten samples test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.chain = helper_mcsamples()

    def test_values(self):
        """Test values."""
        white = stut.whiten_samples(self.chain.samples, self.chain.weights)
        cov = np.cov(white.T, aweights=self.chain.weights)
        self.assertTrue(np.allclose(cov, np.eye(cov.shape[0]), atol=1e-3))

    def test_errors(self):
        """Test error handling."""
        pass

#########################################################################################################


class TestFilterKwargs(unittest.TestCase):

    """Filter kwargs test suite."""
    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_values(self):
        """Test values."""
        res = stut.filter_kwargs(
            {"dict_to_filter": 1, "function_with_kwargs": 2, "extra": 3}, stut.filter_kwargs
        )
        self.assertEqual(res, {"dict_to_filter": 1, "function_with_kwargs": 2})

    def test_errors(self):
        """Test error handling."""
        pass

#########################################################################################################


class TestIsOutlier(unittest.TestCase):

    """Outlier detection test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.chain = helper_mcsamples(ndim=1)

    def test_values(self):
        """Test values."""
        outliers = stut.is_outlier(self.chain.samples)
        self.assertEqual(outliers.shape[0], self.chain.samples.shape[0])
        single = stut.is_outlier(self.chain.samples[0])
        self.assertEqual(single.shape[0], 1)

    def test_errors(self):
        """Test error handling."""
        pass

#########################################################################################################


def make_chain(num_samples=5, dim=2, seed=0):
    """Make a small chain for stats-utility checks.

    :param num_samples: number of samples to generate.
    :param dim: number of parameters.
    :param seed: RNG seed for reproducibility.
    :returns: MCSamples instance.
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(num_samples, dim))
    weights = np.ones(num_samples)
    loglikes = rng.normal(size=num_samples)
    names = [f"x{i}" for i in range(dim)]
    labels = [f"x_{i}" for i in range(dim)]
    return MCSamples(samples=samples, weights=weights, loglikes=loglikes, names=names, labels=labels)


class TestAdditionalStatsUtilities(unittest.TestCase):
    """Additional stats-utilities test suite."""
    def test_from_confidence_scalar(self):
        """Test From confidence scalar."""
        val = stut.from_confidence_to_sigma(0.5)
        self.assertGreater(val, 0.0)

    def test_from_confidence_invalid_high(self):
        """Test From confidence invalid high."""
        with self.assertRaises(ValueError):
            stut.from_confidence_to_sigma(1.2)

    def test_from_sigma_scalar(self):
        """Test From sigma scalar."""
        prob = stut.from_sigma_to_confidence(0.0)
        self.assertAlmostEqual(prob, 0.0)

    def test_from_sigma_invalid_all_negative(self):
        """Test From sigma invalid all negative."""
        with self.assertRaises(ValueError):
            stut.from_sigma_to_confidence(np.array([-1.0, -2.0]))

    def test_from_chi2_to_sigma_large_ratio(self):
        """Test From chi2 to sigma large ratio."""
        res = stut.from_chi2_to_sigma(50.0, 2.0, exact_threshold=5)
        self.assertTrue(np.isfinite(res))

    def test_qr_inverse_identity(self):
        """Test Q R inverse identity."""
        ident = np.eye(3)
        inv = stut.QR_inverse(ident)
        self.assertTrue(np.allclose(inv, ident))

    def test_clopper_pearson_midrange(self):
        """Test Clopper pearson midrange."""
        low, high = stut.clopper_pearson_binomial_trial(2.0, 4.0)
        self.assertLess(low, high)
        self.assertGreaterEqual(low, 0.0)

    def test_make_list_tuple(self):
        """Test Make list tuple."""
        tpl = ("a", "b")
        self.assertEqual(stut.make_list(tpl), tpl)

    def test_make_list_list(self):
        """Test Make list list."""
        lst = ["a"]
        self.assertEqual(stut.make_list(lst), lst)

    def test_vector_pdm_roundtrip_identity(self):
        """Test Vector pdm roundtrip identity."""
        vec = stut.PDM_to_vector(np.eye(2))
        mat = stut.vector_to_PDM(vec)
        self.assertTrue(np.allclose(mat, np.eye(2)))

    def test_vector_pdm_length(self):
        """Test Vector pdm length."""
        mat = np.array([[2.0, 0.5], [0.5, 1.5]])
        vec = stut.PDM_to_vector(mat)
        self.assertEqual(len(vec), 3)

    def test_bernoulli_thin_temperature_zero(self):
        """Test Bernoulli thin temperature zero."""
        np.random.seed(0)
        chain = make_chain()
        thinned = stut.bernoulli_thin(chain.copy(), temperature=0.0, num_repeats=1)
        self.assertEqual(len(thinned.weights), len(chain.weights))

    def test_random_samples_reshuffle_lengths(self):
        """Test Random samples reshuffle lengths."""
        chain = make_chain()
        reshuffled = stut.random_samples_reshuffle(chain.copy())
        self.assertEqual(reshuffled.samples.shape, chain.samples.shape)
        self.assertEqual(reshuffled.weights.shape, chain.weights.shape)

    def test_whiten_samples_diagonal(self):
        """Test Whiten samples diagonal."""
        samples = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
        weights = np.ones(3)
        white = stut.whiten_samples(samples, weights)
        cov = np.cov(white.T, aweights=weights)
        self.assertTrue(np.allclose(cov, np.eye(2), atol=1e-3))

    def test_is_outlier_detects_far_point(self):
        """Test Is outlier detects far point."""
        pts = np.array([[0.0], [0.1], [10.0]])
        flags = stut.is_outlier(pts, thresh=3.5)
        self.assertEqual(flags.sum(), 1)
        self.assertTrue(flags[-1])

    def test_filter_kwargs_subset(self):
        """Test Filter kwargs subset."""
        def dummy(a, b, c=1):
            """Dummy."""
            return a + b + c
        res = stut.filter_kwargs({"a": 1, "b": 2, "c": 3, "d": 4}, dummy)
        self.assertEqual(res, {"a": 1, "b": 2, "c": 3})
        self.assertEqual(dummy(**res), 6)

    def test_kl_decomposition_identity(self):
        """Test K L decomposition identity."""
        lam, vec = stut.KL_decomposition(np.eye(2), np.eye(2))
        self.assertTrue(np.allclose(lam, np.ones(2)))
        self.assertTrue(np.allclose(vec, np.eye(2)))

    def test_random_samples_reshuffle_no_loglikes(self):
        """Test Random samples reshuffle no loglikes."""
        chain = make_chain()
        chain.loglikes = None
        reshuffled = stut.random_samples_reshuffle(chain)
        self.assertEqual(reshuffled.samples.shape[0], chain.samples.shape[0])

    def test_clopper_pearson_near_certainty(self):
        """Test Clopper-Pearson near certainty."""
        low, high = stut.clopper_pearson_binomial_trial(9.0, 10.0)
        self.assertLess(low, 1.0)
        self.assertGreater(high, 0.5)

    def test_from_confidence_to_sigma_array(self):
        """Test confidence-to-sigma array handling."""
        arr = np.array([0.1, 0.9])
        res = stut.from_confidence_to_sigma(arr)
        self.assertEqual(res.shape, arr.shape)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
