"""Additional numerical tests for the fixed (prior) bijectors."""

#########################################################################################################
# Imports

import math
import unittest

import numpy as np
import torch
from scipy import stats

from tensiometer.synthetic_probability import bijectors as bj
from tensiometer.synthetic_probability import fixed_bijectors as fb
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Helper functions


def _tolerance():
    """Absolute and relative tolerance for the active precision.

    :returns: 1e-10 in float64, 1e-5 in float32.
    """
    if tu.get_precision() == torch.float64:
        return 1e-10
    return 1e-5


def _assert_close(actual, expected, tolerance=None):
    """Compare two tensors or arrays on the host.

    :param actual: computed value.
    :param expected: reference value.
    :param tolerance: absolute and relative tolerance, defaults to :func:`_tolerance`.
    """
    if tolerance is None:
        tolerance = _tolerance()
    np.testing.assert_allclose(tu.to_numpy(actual), tu.to_numpy(expected), rtol=tolerance, atol=tolerance)

#########################################################################################################
# One dimensional priors


class TestOneDimensionalPriors(unittest.TestCase):
    """Uniform and normal prior bijectors."""

    def test_uniform_prior_maps_normal_samples_to_uniform(self):
        """Test that uniform_prior maps N(0, 1) samples into [a, b] with a uniform distribution."""
        lower, upper = -1.5, 4.
        bijector = fb.uniform_prior(lower, upper)
        self.assertIsInstance(bijector, bj.Chain)
        torch.manual_seed(0)
        samples = torch.randn(5000, 1, dtype=tu.get_precision())
        mapped = bijector.forward(samples)
        self.assertEqual(mapped.dtype, tu.get_precision())
        mapped = tu.to_numpy(mapped)[:, 0]
        self.assertTrue(np.all(mapped >= lower))
        self.assertTrue(np.all(mapped <= upper))
        result = stats.kstest(mapped, stats.uniform(loc=lower, scale=upper - lower).cdf)
        self.assertGreater(result.pvalue, 0.01)

    def test_uniform_prior_values_and_log_det(self):
        """Test uniform_prior against the normal CDF and its round trip."""
        lower, upper = 0., 2.
        bijector = fb.uniform_prior(lower, upper)
        x = np.linspace(-2., 2., 9)[:, None]
        expected = lower + (upper - lower) * stats.norm.cdf(x)
        _assert_close(bijector.forward(x), expected)
        _assert_close(bijector.inverse(expected), x, tolerance=max(_tolerance(), 1e-4))
        _assert_close(bijector.forward_log_det_jacobian(x, event_ndims=0),
                      np.log(upper - lower) + stats.norm.logpdf(x))
        _assert_close(bijector.forward_log_det_jacobian(x, event_ndims=1),
                      np.log(upper - lower) + stats.norm.logpdf(x[:, 0]))

    def test_normal(self):
        """Test that normal(mean, sigma) is the affine map mean + sigma x."""
        bijector = fb.normal(mean=2., sigma=0.5)
        x = np.linspace(-3., 3., 7)[:, None]
        _assert_close(bijector.forward(x), 2. + 0.5 * x)
        _assert_close(bijector.inverse(2. + 0.5 * x), x)
        _assert_close(bijector.forward_log_det_jacobian(x), np.full(7, math.log(0.5)))

    def test_normal_accepts_tensors(self):
        """Test that normal and uniform_prior accept tensor arguments."""
        x = np.array([[0.3]])
        _assert_close(fb.normal(torch.tensor(1.), torch.tensor(2.)).forward(x), fb.normal(1., 2.).forward(x))
        _assert_close(fb.uniform_prior(torch.tensor(0.), torch.tensor(1.)).forward(x),
                      fb.uniform_prior(0., 1.).forward(x))

#########################################################################################################
# Multivariate normal


class TestMultivariateNormal(unittest.TestCase):
    """Multivariate normal bijector."""

    def setUp(self):
        """Mean and covariance of a correlated Gaussian."""
        self.mean = np.array([0.5, -1., 2.])
        self.covariance = np.array([[1., 0.4, -0.2], [0.4, 2., 0.3], [-0.2, 0.3, 0.5]])
        self.bijector = fb.multivariate_normal(self.mean, self.covariance)

    def test_type_and_dtype(self):
        """Test the bijector type and the dtype of its buffers."""
        self.assertIsInstance(self.bijector, bj.AffineTriL)
        self.assertEqual(self.bijector.scale_tril.dtype, tu.get_precision())
        self.assertEqual(self.bijector.shift.dtype, tu.get_precision())

    def test_forward_of_zero_is_mean(self):
        """Test that the origin maps to the mean."""
        _assert_close(self.bijector.forward(np.zeros((1, 3))), self.mean[None, :])

    def test_forward_matches_cholesky(self):
        """Test the forward map against the numpy Cholesky factor."""
        cholesky = np.linalg.cholesky(self.covariance)
        x = np.random.default_rng(0).standard_normal((5, 3))
        _assert_close(self.bijector.forward(x), x @ cholesky.T + self.mean)

    def test_inverse_whitens_samples(self):
        """Test that the inverse maps samples of the Gaussian to unit covariance."""
        random_state = np.random.default_rng(1)
        samples = random_state.multivariate_normal(self.mean, self.covariance, size=20000)
        whitened = tu.to_numpy(self.bijector.inverse(samples))
        np.testing.assert_allclose(whitened.mean(axis=0), np.zeros(3), atol=0.03)
        np.testing.assert_allclose(np.cov(whitened.T), np.eye(3), atol=0.04)

    def test_log_det_is_sum_log_diag_cholesky(self):
        """Test fldj == sum log diag chol and ildj == -fldj."""
        expected = np.sum(np.log(np.diag(np.linalg.cholesky(self.covariance))))
        x = np.random.default_rng(2).standard_normal((4, 3))
        _assert_close(self.bijector.forward_log_det_jacobian(x), np.full(4, expected))
        _assert_close(self.bijector.inverse_log_det_jacobian(self.bijector.forward(x)), np.full(4, -expected))
        _assert_close(expected, 0.5 * np.linalg.slogdet(self.covariance)[1])

#########################################################################################################
# Prior helper


class TestPriorBijectorHelper(unittest.TestCase):
    """Composite prior bijector helper."""

    def test_mixed_entries(self):
        """Test a list of uniform, Gaussian and None entries."""
        priors = [
            {'mode': 'uniform', 'lower': 0., 'upper': 1.},
            {'mode': 'gaussian', 'mean': 1., 'scale': 2.},
            None,
        ]
        bijector = fb.prior_bijector_helper(prior_dict_list=priors, name='combo')
        self.assertIsInstance(bijector, bj.Blockwise)
        self.assertEqual(bijector.name, 'combo')
        self.assertEqual(bijector.block_sizes, [1, 1, 1])
        self.assertIsInstance(bijector.bijectors[2], bj.Identity)
        x = np.random.default_rng(3).standard_normal((6, 3))
        expected = np.stack([stats.norm.cdf(x[:, 0]), 1. + 2. * x[:, 1], x[:, 2]], axis=-1)
        _assert_close(bijector.forward(x), expected)
        _assert_close(bijector.inverse(expected), x, tolerance=max(_tolerance(), 1e-4))
        expected_log_det = stats.norm.logpdf(x[:, 0]) + math.log(2.)
        _assert_close(bijector.forward_log_det_jacobian(x), expected_log_det)

    def test_all_none_entries(self):
        """Test that None entries give the identity."""
        bijector = fb.prior_bijector_helper(prior_dict_list=[None, None])
        x = np.array([[0.3, -0.4]])
        _assert_close(bijector.forward(x), x)
        _assert_close(bijector.forward_log_det_jacobian(x), np.zeros(1), tolerance=0.)

    def test_multivariate(self):
        """Test the multivariate Gaussian branch."""
        loc = np.array([1., 2.])
        cov = np.array([[2., 0.5], [0.5, 1.]])
        bijector = fb.prior_bijector_helper(loc=loc, cov=cov)
        self.assertIsInstance(bijector, bj.AffineTriL)
        _assert_close(bijector.forward(np.zeros((1, 2))), loc[None, :])

    def test_errors(self):
        """Test prior_bijector_helper error handling."""
        with self.assertRaises(ValueError):
            fb.prior_bijector_helper(prior_dict_list=[{'mode': 'unknown'}])
        with self.assertRaises(ValueError):
            fb.prior_bijector_helper(prior_dict_list=[{'lower': 0., 'upper': 1.}])
        with self.assertRaises(ValueError):
            fb.prior_bijector_helper()
        with self.assertRaises(ValueError):
            fb.prior_bijector_helper(loc=np.zeros(2))

#########################################################################################################
# Modulus bijector


class TestMod1D(unittest.TestCase):
    """Modulus bijector."""

    def test_wraps_into_range(self):
        """Test that Mod1D wraps values into [minval, maxval)."""
        bijector = fb.Mod1D(minval=-1., maxval=1.)
        x = np.array([-2.5, -1., -0.5, 0.5, 1.5, 3.25])
        y = tu.to_numpy(bijector.forward(x))
        _assert_close(y, np.array([-0.5, -1., -0.5, 0.5, -0.5, -0.75]))
        self.assertTrue(np.all(y >= -1.))
        self.assertTrue(np.all(y < 1.))

    def test_periodicity_and_inverse(self):
        """Test periodicity, idempotence and the inverse."""
        bijector = fb.Mod1D(minval=0., maxval=2. * math.pi, name='angle')
        self.assertEqual(bijector.name, 'angle')
        x = np.random.default_rng(4).uniform(0., 2. * math.pi, 20)
        _assert_close(bijector.forward(x + 4. * math.pi), x, tolerance=max(_tolerance(), 1e-5))
        _assert_close(bijector.forward(bijector.forward(x)), bijector.forward(x), tolerance=0.)
        _assert_close(bijector.inverse(x - 2. * math.pi), x, tolerance=max(_tolerance(), 1e-5))

    def test_zero_log_det(self):
        """Test that the log determinants vanish."""
        bijector = fb.Mod1D(minval=-1., maxval=1.)
        x = np.array([[-2.5, -0.5], [0.5, 1.5]])
        _assert_close(bijector.forward_log_det_jacobian(x, event_ndims=0), np.zeros((2, 2)), tolerance=0.)
        _assert_close(bijector.inverse_log_det_jacobian(x, event_ndims=1), np.zeros(2), tolerance=0.)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
