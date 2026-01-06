"""Additional tests for fixed bijectors."""

#########################################################################################################
# Imports

import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensiometer.synthetic_probability import fixed_bijectors as fb

#########################################################################################################
# TensorFlow Probability aliases

tfb = tfp.bijectors

#########################################################################################################
# Test cases


class TestFixedBijectorsAdditional(unittest.TestCase):
    """Fixed bijectors additional test suite."""
    def test_uniform_prior_chain(self):
        """Test uniform prior bijector."""
        bij = fb.uniform_prior(0.0, 1.0)
        x = tf.constant([0.25, 0.75])
        y = bij.forward(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(bij.forward_log_det_jacobian(x, event_ndims=0))))

    def test_normal_bijector(self):
        """Test normal bijector construction."""
        bij = fb.normal(mean=2.0, sigma=0.5)
        x = tf.constant([0.0, 1.0])
        y = bij.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_multivariate_normal_bijector(self):
        """Test multivariate normal bijector."""
        mean = np.zeros(2, dtype=np.float32)
        cov = np.eye(2, dtype=np.float32)
        dist = tfp.distributions.MultivariateNormalTriL(
            loc=mean, scale_tril=tf.linalg.cholesky(cov)
        )
        bij = dist.bijector
        x = tf.constant([[0.0, 1.0]], dtype=tf.float32)
        y = bij.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_prior_bijector_helper_uniform_and_gaussian(self):
        """Test prior bijector helper with uniform and gaussian specs."""
        priors = [
            {"mode": "uniform", "lower": 0.0, "upper": 1.0},
            {"mode": "gaussian", "mean": 0.0, "scale": 1.0},
            None,
        ]
        bij = fb.prior_bijector_helper(prior_dict_list=priors, name="combo")
        x = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)
        y = bij.forward(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(bij.name, "combo")

    def test_prior_bijector_helper_multivariate(self):
        """Test prior bijector helper with multivariate inputs."""
        loc = np.zeros(2, dtype=np.float32)
        cov = np.eye(2, dtype=np.float32)
        # Monkeypatch helper to return a known bijector
        orig = fb.multivariate_normal
        fb.multivariate_normal = lambda mean, covariance: tfp.distributions.MultivariateNormalTriL(
            loc=mean, scale_tril=tf.linalg.cholesky(covariance)
        ).bijector
        bij = fb.prior_bijector_helper(loc=loc, cov=cov)
        x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        y = bij.forward(x)
        self.assertEqual(y.shape, x.shape)
        fb.multivariate_normal = orig

    def test_prior_bijector_helper_errors(self):
        """Test prior bijector helper error handling."""
        with self.assertRaises(ValueError):
            fb.prior_bijector_helper(prior_dict_list=[{"mode": "unknown"}])
        with self.assertRaises(ValueError):
            fb.prior_bijector_helper()
        with self.assertRaises(AssertionError):
            fb.prior_bijector_helper(loc=np.zeros(1, dtype=np.float32))

    def test_prior_bijector_helper_missing_mode_entry(self):
        """Test prior bijector helper with missing mode entry."""
        priors = [{"lower": 0.0, "upper": 1.0}]
        with self.assertRaises(Exception):
            fb.prior_bijector_helper(prior_dict_list=priors)

    def test_mod1d_properties(self):
        """Test Mod1D bijector properties."""
        bij = fb.Mod1D(minval=-1.0, maxval=1.0, dtype=tf.float32)
        x = tf.constant([-2.5, -0.5, 0.5, 1.5], dtype=tf.float32)
        y = bij.forward(x)
        self.assertTrue(tf.reduce_all(y <= 1.0))
        self.assertTrue(tf.reduce_all(y >= -1.0))
        inv = bij.inverse(y)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(inv)))
        fldj = bij.forward_log_det_jacobian(x, event_ndims=0)
        ildj = bij.inverse_log_det_jacobian(y, event_ndims=0)
        self.assertTrue(tf.reduce_all(fldj == 0.0))
        self.assertTrue(tf.reduce_all(ildj == 0.0))
        self.assertTrue(bij._is_increasing())

        direct_inv = bij._inverse(y)
        direct_ildj = bij._inverse_log_det_jacobian(y)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(direct_inv)))
        self.assertTrue(tf.reduce_all(direct_ildj == 0.0))

    def test_multivariate_normal_helper(self):
        """Test multivariate_normal helper."""
        mean = np.array([0.0, 1.0], dtype=np.float32)
        cov = np.array([[1.0, 0.1], [0.1, 2.0]], dtype=np.float32)

        class DummyMVN:
            """Dummy MVN test suite."""
            def __init__(self, mean=None, scale_tril=None, **kwargs):
                """Init."""
                self.args = {"mean": mean, "scale_tril": scale_tril}
                self.bijector = tfb.Identity()

        orig = fb.tfd.MultivariateNormalTriL
        fb.tfd.MultivariateNormalTriL = DummyMVN
        try:
            bij = fb.multivariate_normal(mean, cov)
        finally:
            fb.tfd.MultivariateNormalTriL = orig

        x = tf.constant([[0.5, -0.5]], dtype=tf.float32)
        y = bij.forward(x)
        self.assertEqual(y.shape, x.shape)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
