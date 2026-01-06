"""Tests for analytic flow utilities and wrappers."""

#########################################################################################################
# Imports

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensiometer.synthetic_probability import analytic_flow as af

#########################################################################################################
# TensorFlow Probability aliases

tfd = tfp.distributions

#########################################################################################################
# Helper distributions


class DummyDistribution:
    """Simple Gaussian-like distribution for exercising analytic_flow."""

    def __init__(self, with_log=True):
        """Initialize the dummy distribution.

        :param with_log: whether to attach a ``log_pdf`` attribute.
        """
        self.label = "dummy"
        self.names = ["p0", "p1"]
        self.labels = ["p0", "p1"]
        self.lims = {"p0": [-1.0, 1.0], "p1": [-1.0, 1.0]}
        self.with_log = with_log
        if with_log:
            self.log_pdf = self._log_pdf

    def sim(self, n):
        """Draw samples from the dummy distribution.

        :param n: number of samples to draw.
        :returns: samples with shape ``(n, 2)``.
        """
        rng = np.random.default_rng(42)
        return rng.normal(scale=0.5, size=(n, 2))

    def pdf(self, x):
        """Evaluate the probability density.

        :param x: sample coordinates.
        :returns: density values for ``x``.
        """
        arr = np.asarray(x)
        return np.exp(-0.5 * np.sum(arr**2, axis=-1)) / (2 * np.pi)

    def _log_pdf(self, x):
        """Evaluate the log-probability density.

        :param x: sample coordinates.
        :returns: log-density values for ``x``.
        """
        arr = np.asarray(x)
        return -0.5 * np.sum(arr**2, axis=-1) - np.log(2 * np.pi)


#########################################################################################################
# Wrapper tests


class TestTfProbWrapper(unittest.TestCase):
    """TensorFlow Probability wrapper test suite."""
    def test_wrapper_uses_distribution(self):
        """Test wrapper uses distribution metadata."""
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 2.0], name="mvdiag"
        )
        wrapper = af.tf_prob_wrapper(dist)
        coords = tf.constant([[0.5, -0.5]], dtype=tf.float32)
        logp = wrapper.log_pdf(coords)
        self.assertEqual(logp.shape, (1,))
        samples = wrapper.sim(3)
        self.assertEqual(samples.shape, (3, 2))
        self.assertEqual(wrapper.names, ["p0", "p1"])


#########################################################################################################
# Analytic flow tests


class TestAnalyticFlow(unittest.TestCase):
    """Analytic flow test suite."""
    def test_invalid_inputs_raise(self):
        """Test invalid input validation."""
        class MissingSim:
            """Missing Sim test suite."""
            def pdf(self, x):
                """Pdf."""
                return x

        class MissingPdf:
            """Missing Pdf test suite."""
            def sim(self, n):
                """Sim."""
                return np.zeros((n, 1))

        with self.assertRaises(ValueError):
            af.analytic_flow(MissingSim())
        with self.assertRaises(ValueError):
            af.analytic_flow(MissingPdf())
        self.assertTrue(np.allclose(MissingSim().pdf(np.array([1.0])), np.array([1.0])))
        self.assertEqual(MissingPdf().sim(1).shape, (1, 1))

    def test_log_probability_and_derivatives(self):
        """Test log-probability values and derivatives."""
        flow = af.analytic_flow(DummyDistribution(with_log=True))
        test_point = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        lp = flow.log_probability(test_point)
        self.assertTrue(np.all(np.isfinite(lp.numpy())))
        jac = flow.log_probability_jacobian(np.array([0.1, -0.1]))
        self.assertEqual(jac.shape, (2,))
        hess = flow.log_probability_hessian(np.array([0.2, -0.2]))
        self.assertEqual(hess.shape, (2, 2))
        samples = flow.sample(4)
        self.assertEqual(samples.shape, (4, 2))
        self.assertIsNone(flow.reset_tensorflow_caches())

    def test_branch_without_log_pdf(self):
        """Test branch that falls back to pdf without log_pdf."""
        flow = af.analytic_flow(DummyDistribution(with_log=False))
        coords = np.array([[0.1, 0.2], [0.0, 0.0]])
        lp = flow.log_probability(coords).numpy()
        manual = np.log(flow._dist.pdf(coords))
        np.testing.assert_allclose(lp, manual)
        jac = flow.log_probability_jacobian(coords)
        self.assertEqual(jac.shape, coords.shape)

    def test_mcsamples_output(self):
        """Test MCSamples output shape and metadata."""
        flow = af.analytic_flow(DummyDistribution())
        mc = flow.MCSamples(6)
        self.assertEqual(mc.samples.shape[0], 6)
        self.assertEqual(len(mc.loglikes), 6)
        running = mc.getParamNames().getRunningNames()
        self.assertEqual(set(running), {"p0", "p1"})

#########################################################################################################
# Additional helper distributions


class DummyNumpyDist:
    """Dummy Numpy Dist test suite."""
    def __init__(self):
        """Init."""
        self.label = "dummy"
        self.names = ["x"]
        self.labels = ["x"]
        self.lims = {"x": (0.0, 1.0)}

    def sim(self, n):
        """Simulate samples.

        :param n: number of samples.
        :returns: samples with shape ``(n, 1)``.
        """
        return np.ones((n, 1))

    def pdf(self, x):
        """Evaluate the probability density.

        :param x: sample coordinates.
        :returns: density values for ``x``.
        """
        return np.ones_like(x)


class DummyNumpyLogPDF(DummyNumpyDist):
    """Dummy Numpy Log P D F test suite."""
    def log_pdf(self, x):
        """Evaluate the log density.

        :param x: sample coordinates.
        :returns: log-density values for ``x``.
        """
        return np.zeros_like(x)

#########################################################################################################
# Additional analytic flow tests


class TestAnalyticFlowAdditional(unittest.TestCase):
    """Analytic flow additional test suite."""
    def test_tf_prob_wrapper(self):
        """Test TensorFlow Probability wrapper precisions."""
        dist = tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 2.0])
        wrapper = af.tf_prob_wrapper(dist, prec=tf.float32)
        samp = wrapper.sim(2)
        self.assertEqual(samp.shape[-1], 2)
        lp = wrapper.log_pdf(tf.constant([[0.0, 0.0]], dtype=tf.float32))
        self.assertEqual(lp.shape, (1,))

    def test_init_requires_pdf(self):
        """Test initialization requires pdf attribute."""
        class BadDist:
            """Bad Dist test suite."""
            def sim(self, n):
                """Simulate samples.

                :param n: number of samples.
                :returns: samples with shape ``(n, 1)``.
                """
                return np.zeros((n, 1))
        self.assertEqual(BadDist().sim(2).shape, (2, 1))
        with self.assertRaises(ValueError):
            af.analytic_flow(BadDist())

    def test_init_without_log_pdf(self):
        """Test initialization without log_pdf attribute."""
        dist = DummyNumpyDist()
        flow = af.analytic_flow(dist)
        self.assertFalse(flow.has_log_pdf)

    def test_log_probability_branches(self):
        """Test log_probability branches with and without log_pdf."""
        flow_with_log = af.analytic_flow(DummyNumpyLogPDF())
        flow_without_log = af.analytic_flow(DummyNumpyDist())
        coords = np.array([[0.0], [0.5]])
        res_with = flow_with_log.log_probability(coords)
        res_without = flow_without_log.log_probability(coords)
        self.assertTrue(np.allclose(res_with.numpy(), 0.0))
        self.assertTrue(np.allclose(res_without.numpy(), 0.0))

    def test_log_probability_tensor_input(self):
        """Test tensor input handling in log_probability."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        coords = tf.constant([[0.1], [0.2]], dtype=tf.float32)
        res = flow.log_probability(coords)
        self.assertEqual(res.shape[0], 2)

    def test_jacobian_and_hessian(self):
        """Test Jacobian and Hessian calculations."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        coords = np.array([[0.1], [0.2]])
        jac = flow.log_probability_jacobian(coords)
        self.assertEqual(jac.shape[0], 2)
        hess = flow.log_probability_hessian(coords)
        self.assertEqual(hess.shape[0], 2)
        jac_scalar = flow.log_probability_jacobian(coords[0])
        hess_scalar = flow.log_probability_hessian(coords[0])
        self.assertEqual(jac_scalar.shape[0], 1)
        self.assertEqual(hess_scalar.shape[0], 1)

    def test_mcsamples_generation(self):
        """Test MCSamples generation with log-likelihoods."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        mc = flow.MCSamples(size=3, logLikes=True)
        self.assertEqual(mc.samples.shape[0], 3)
        self.assertIsNotNone(mc.loglikes)

    def test_reset_placeholder(self):
        """Test reset_tensorflow_caches placeholder."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        self.assertIsNone(flow.reset_tensorflow_caches())

    def test_custom_metadata_overrides_defaults(self):
        """Test metadata overrides for analytic_flow inputs."""
        class CustomDist(DummyNumpyLogPDF):
            """Custom Dist test suite."""
            def __init__(self):
                """Init."""
                super().__init__()
                self.label = "base"
                self.names = ["a"]
                self.labels = ["a"]
                self.lims = {"a": (-1.0, 1.0)}

        dist = CustomDist()
        flow = af.analytic_flow(
            dist,
            name_tag="override",
            param_names=["x"],
            param_labels=["x_label"],
            lims={"x": (0.0, 2.0)},
        )
        self.assertEqual(flow.name_tag, "override")
        self.assertEqual(flow.param_names, ["x"])
        self.assertEqual(flow.param_labels, ["x_label"])
        self.assertEqual(flow.parameter_ranges, {"x": (0.0, 2.0)})

    def test_mcsamples_without_loglikes(self):
        """Test MCSamples path without loglikes."""
        flow = af.analytic_flow(DummyNumpyLogPDF())

        class DummyMCSamples:
            """Dummy MCSamples test suite."""
            def __init__(self, samples, loglikes, **kwargs):
                """Init."""
                self.samples = samples
                self.loglikes = loglikes

        dummy = DummyMCSamples(np.zeros((1, 1)), None)
        self.assertEqual(dummy.samples.shape, (1, 1))

        with patch("tensiometer.synthetic_probability.analytic_flow.MCSamples", DummyMCSamples):
            with self.assertRaises(AttributeError):
                flow.MCSamples(size=2, logLikes=False)

    def test_jacobian_tensor_without_log_pdf(self):
        """Test tensor Jacobian without log_pdf."""
        flow = af.analytic_flow(DummyNumpyDist())
        coords = tf.constant([0.5], dtype=tf.float32)
        jac = flow.log_probability_jacobian(coords)
        self.assertEqual(jac.shape, (1,))

    def test_hessian_tensor_input(self):
        """Test tensor Hessian input handling."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        coords = tf.constant([[0.0]], dtype=tf.float32)
        hess = flow.log_probability_hessian(coords)
        self.assertEqual(hess.shape[0], 1)

    def test_tensor_jacobian_and_hessian_branches(self):
        """Test tensor Jacobian and Hessian branches."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        tensor_coords = tf.constant([[0.0]], dtype=tf.float32)
        jac = flow.log_probability_jacobian(tensor_coords)
        hess = flow.log_probability_hessian(tensor_coords)
        self.assertEqual(jac.shape[0], 1)
        self.assertEqual(hess.shape[0], 1)

    def test_scalar_jacobian_without_log_pdf(self):
        """Test scalar Jacobian without log_pdf."""
        flow = af.analytic_flow(DummyNumpyDist())
        jac = flow.log_probability_jacobian(np.array([0.5]))
        self.assertEqual(jac.shape, (1,))


#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
