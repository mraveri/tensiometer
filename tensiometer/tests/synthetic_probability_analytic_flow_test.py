"""Tests for analytic flow utilities and wrappers."""

#########################################################################################################
# Imports

import unittest
from unittest.mock import patch

import numpy as np
import scipy.stats
import torch

from tensiometer.synthetic_probability import analytic_flow as af
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Helper functions


def _diagonal_gaussian(loc, scale):
    """Build a torch multivariate normal with diagonal covariance and the active precision.

    :param loc: mean vector, shape ``(D,)``.
    :param scale: standard deviations, shape ``(D,)``.
    :returns: ``torch.distributions.MultivariateNormal`` instance.
    """
    loc = torch.as_tensor(loc, dtype=tu.get_precision())
    scale = torch.as_tensor(scale, dtype=tu.get_precision())
    return torch.distributions.MultivariateNormal(loc, scale_tril=torch.diag(scale))

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


class TestTorchProbWrapper(unittest.TestCase):
    """Torch distribution wrapper test suite."""
    def test_wrapper_uses_distribution(self):
        """Test wrapper uses distribution metadata."""
        dist = _diagonal_gaussian([0.0, 0.0], [1.0, 2.0])
        wrapper = af.torch_prob_wrapper(dist, name="mvdiag")
        coords = torch.tensor([[0.5, -0.5]], dtype=tu.get_precision())
        logp = wrapper.log_pdf(coords)
        self.assertEqual(tuple(logp.shape), (1,))
        samples = wrapper.sim(3)
        self.assertEqual(tuple(samples.shape), (3, 2))
        self.assertEqual(wrapper.names, ["p0", "p1"])
        self.assertEqual(wrapper.labels, ["p0", "p1"])
        self.assertIsNone(wrapper.lims)
        self.assertEqual(wrapper.label, "mvdiag")
        self.assertEqual(wrapper.num_params, 2)

    def test_wrapper_default_name_and_outputs(self):
        """Test default label, output devices and dtypes."""
        wrapper = af.torch_prob_wrapper(_diagonal_gaussian([0.0], [1.0]))
        self.assertEqual(wrapper.label, "MultivariateNormal")
        self.assertEqual(wrapper.prec, tu.get_precision())
        logp = wrapper.log_pdf(np.zeros((2, 1)))
        self.assertTrue(torch.is_tensor(logp))
        self.assertEqual(logp.device.type, "cpu")
        self.assertEqual(logp.dtype, tu.get_precision())
        self.assertEqual(wrapper.sim(4).device.type, "cpu")

    def test_wrapper_values_match_scipy(self):
        """Test log_pdf and pdf against scipy."""
        loc = np.array([0.5, -1.0])
        scale = np.array([1.0, 2.0])
        wrapper = af.torch_prob_wrapper(_diagonal_gaussian(loc, scale))
        coords = np.array([[0.0, 0.0], [1.0, -2.0], [0.5, 3.0]])
        reference = scipy.stats.multivariate_normal(loc, np.diag(scale**2)).logpdf(coords)
        np.testing.assert_allclose(tu.to_numpy(wrapper.log_pdf(coords)), reference, rtol=1e-5)
        np.testing.assert_allclose(wrapper.pdf(coords), np.exp(reference), rtol=1e-5)

    def test_analytic_flow_from_wrapper(self):
        """Test an analytic flow built on a wrapped torch distribution."""
        torch.manual_seed(0)
        loc = np.array([0.0, 1.0])
        scale = np.array([1.0, 0.5])
        wrapper = af.torch_prob_wrapper(_diagonal_gaussian(loc, scale), name="gauss")
        flow = af.analytic_flow(wrapper)
        self.assertEqual(flow.name_tag, "gauss")
        self.assertEqual(flow.num_params, 2)
        self.assertIsNone(flow.parameter_ranges)
        coords = np.array([[0.1, 0.9], [-0.3, 1.2]])
        reference = scipy.stats.multivariate_normal(loc, np.diag(scale**2)).logpdf(coords)
        np.testing.assert_allclose(tu.to_numpy(flow.log_probability(coords)), reference, rtol=1e-5)
        mc = flow.MCSamples(50)
        self.assertEqual(mc.samples.shape, (50, 2))
        self.assertEqual(mc.getParamNames().getRunningNames(), ["p0", "p1"])

    def test_analytic_flow_derivatives_from_float64_wrapper(self):
        """Test finite difference derivatives of a wrapped float64 distribution.

        The derivatives use scipy finite differences with steps of order ``1e-8``, below the
        float32 resolution, so the wrapped distribution is evaluated in float64.
        """
        loc = np.array([0.0, 1.0])
        scale = np.array([1.0, 0.5])
        dist = torch.distributions.MultivariateNormal(
            torch.as_tensor(loc, dtype=torch.float64),
            scale_tril=torch.diag(torch.as_tensor(scale, dtype=torch.float64)))
        flow = af.analytic_flow(af.torch_prob_wrapper(dist, prec=torch.float64))
        coords = np.array([[0.1, 0.9], [-0.3, 1.2]])
        jac = tu.to_numpy(flow.log_probability_jacobian(coords))
        np.testing.assert_allclose(jac, -(coords - loc) / scale**2, atol=1e-4)
        hess = tu.to_numpy(flow.log_probability_hessian(coords[0]))
        np.testing.assert_allclose(hess, -np.diag(1.0 / scale**2), atol=1e-4)


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
        with self.assertRaises(ValueError):
            af.analytic_flow(DummyDistribution(), param_names=["a", "b", "c"])
        self.assertTrue(np.allclose(MissingSim().pdf(np.array([1.0])), np.array([1.0])))
        self.assertEqual(MissingPdf().sim(1).shape, (1, 1))

    def test_log_probability_and_derivatives(self):
        """Test log-probability values and derivatives."""
        flow = af.analytic_flow(DummyDistribution(with_log=True))
        test_point = torch.tensor([[0.0, 0.0]], dtype=tu.get_precision())
        lp = flow.log_probability(test_point)
        self.assertTrue(np.all(np.isfinite(lp.numpy())))
        self.assertEqual(lp.dtype, tu.get_precision())
        jac = flow.log_probability_jacobian(np.array([0.1, -0.1]))
        self.assertEqual(tuple(jac.shape), (2,))
        hess = flow.log_probability_hessian(np.array([0.2, -0.2]))
        self.assertEqual(tuple(hess.shape), (2, 2))
        np.testing.assert_allclose(tu.to_numpy(hess), -np.eye(2), atol=1e-4)
        samples = flow.sample(4)
        self.assertEqual(tuple(samples.shape), (4, 2))
        self.assertTrue(torch.is_tensor(samples))

    def test_num_params_from_param_names(self):
        """num_params follows param_names, also for distributions without ``names``."""
        class NoNames(DummyDistribution):
            """Dummy distribution without names."""
            def __init__(self):
                """Drop the names attribute."""
                super().__init__()
                del self.names

        flow = af.analytic_flow(NoNames(), param_names=["a", "b"])
        self.assertEqual(flow.num_params, 2)
        self.assertEqual(flow.param_names, ["a", "b"])
        flow = af.analytic_flow(DummyDistribution(), param_names=["a", "b"])
        self.assertEqual(flow.num_params, 2)

    def test_branch_without_log_pdf(self):
        """Test branch that falls back to pdf without log_pdf."""
        flow = af.analytic_flow(DummyDistribution(with_log=False))
        coords = np.array([[0.1, 0.2], [0.0, 0.0]])
        lp = flow.log_probability(coords).numpy()
        manual = np.log(flow._dist.pdf(coords))
        np.testing.assert_allclose(lp, manual, rtol=1e-6)
        jac = flow.log_probability_jacobian(coords)
        self.assertEqual(tuple(jac.shape), coords.shape)

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
    def test_torch_prob_wrapper(self):
        """Test torch distribution wrapper with an explicit precision."""
        dist = _diagonal_gaussian([0.0, 0.0], [1.0, 2.0])
        wrapper = af.torch_prob_wrapper(dist, prec=tu.get_precision())
        samp = wrapper.sim(2)
        self.assertEqual(samp.shape[-1], 2)
        lp = wrapper.log_pdf(torch.tensor([[0.0, 0.0]], dtype=tu.get_precision()))
        self.assertEqual(tuple(lp.shape), (1,))

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
        coords = torch.tensor([[0.1], [0.2]], dtype=tu.get_precision())
        res = flow.log_probability(coords)
        self.assertEqual(res.shape[0], 2)

    def test_jacobian_and_hessian(self):
        """Test Jacobian and Hessian calculations."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        coords = np.array([[0.1], [0.2]])
        jac = flow.log_probability_jacobian(coords)
        self.assertEqual(jac.shape[0], 2)
        hess = flow.log_probability_hessian(coords)
        self.assertEqual(tuple(hess.shape), (2, 1, 1))
        np.testing.assert_allclose(tu.to_numpy(hess), 0.0, atol=1e-6)
        jac_scalar = flow.log_probability_jacobian(coords[0])
        hess_scalar = flow.log_probability_hessian(coords[0])
        self.assertEqual(jac_scalar.shape[0], 1)
        self.assertEqual(tuple(hess_scalar.shape), (1, 1))

    def test_mcsamples_generation(self):
        """Test MCSamples generation with log-likelihoods."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        mc = flow.MCSamples(size=3, logLikes=True)
        self.assertEqual(mc.samples.shape[0], 3)
        self.assertIsNotNone(mc.loglikes)

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
        flow = af.analytic_flow(DummyDistribution())
        mc = flow.MCSamples(size=10, logLikes=False)
        self.assertEqual(mc.samples.shape, (10, 2))
        self.assertIsNone(mc.loglikes)

    def test_mcsamples_without_loglikes_arguments(self):
        """Test that MCSamples receives no loglikes and the flow metadata."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        captured = {}

        class DummyMCSamples:
            """Stand-in recording the MCSamples arguments."""
            def __init__(self, samples, loglikes, **kwargs):
                """Init."""
                captured["samples"] = samples
                captured["loglikes"] = loglikes
                captured.update(kwargs)

        with patch("tensiometer.synthetic_probability.analytic_flow.MCSamples", DummyMCSamples):
            flow.MCSamples(size=2, logLikes=False)
        self.assertIsInstance(captured["samples"], np.ndarray)
        self.assertEqual(captured["samples"].shape, (2, 1))
        self.assertIsNone(captured["loglikes"])
        self.assertEqual(captured["names"], ["x"])
        self.assertEqual(captured["name_tag"], "dummy")

    def test_jacobian_tensor_without_log_pdf(self):
        """Test tensor Jacobian without log_pdf."""
        flow = af.analytic_flow(DummyNumpyDist())
        coords = torch.tensor([0.5], dtype=tu.get_precision())
        jac = flow.log_probability_jacobian(coords)
        self.assertEqual(tuple(jac.shape), (1,))

    def test_hessian_tensor_input(self):
        """Test tensor Hessian input handling."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        coords = torch.tensor([[0.0]], dtype=tu.get_precision())
        hess = flow.log_probability_hessian(coords)
        self.assertEqual(tuple(hess.shape), (1, 1, 1))

    def test_tensor_jacobian_and_hessian_branches(self):
        """Test tensor Jacobian and Hessian branches."""
        flow = af.analytic_flow(DummyNumpyLogPDF())
        tensor_coords = torch.tensor([[0.0]], dtype=tu.get_precision())
        jac = flow.log_probability_jacobian(tensor_coords)
        hess = flow.log_probability_hessian(tensor_coords)
        self.assertEqual(jac.shape[0], 1)
        self.assertEqual(tuple(hess.shape), (1, 1, 1))

    def test_jacobian_with_scalar_log_pdf(self):
        """Test the log_pdf Jacobian branch with a log_pdf returning scalars."""
        class ScalarLogPDF(DummyNumpyDist):
            """One dimensional Gaussian with a scalar log_pdf."""
            def log_pdf(self, x):
                """Evaluate the log density."""
                return -0.5 * np.sum(np.asarray(x)**2, axis=-1)

        flow = af.analytic_flow(ScalarLogPDF())
        coords = np.array([[0.1], [0.2]])
        jac = flow.log_probability_jacobian(coords)
        np.testing.assert_allclose(tu.to_numpy(jac), -coords, atol=1e-4)
        jac_tensor = flow.log_probability_jacobian(torch.tensor([[0.3]], dtype=tu.get_precision()))
        self.assertEqual(tuple(jac_tensor.shape), (1, 1))
        jac_scalar = flow.log_probability_jacobian(coords[0])
        np.testing.assert_allclose(tu.to_numpy(jac_scalar), [-0.1], atol=1e-4)
        hess = flow.log_probability_hessian(coords)
        self.assertEqual(tuple(hess.shape), (2, 1, 1))
        np.testing.assert_allclose(tu.to_numpy(hess), -1.0, atol=1e-4)
        hess_scalar = flow.log_probability_hessian(coords[0])
        self.assertEqual(tuple(hess_scalar.shape), (1, 1))
        np.testing.assert_allclose(tu.to_numpy(hess_scalar), -1.0, atol=1e-4)

    def test_scalar_jacobian_without_log_pdf(self):
        """Test scalar Jacobian without log_pdf."""
        flow = af.analytic_flow(DummyNumpyDist())
        jac = flow.log_probability_jacobian(np.array([0.5]))
        self.assertEqual(tuple(jac.shape), (1,))


#########################################################################################################
# Script entry point


#########################################################################################################
# Finite difference Hessian


class TestFiniteDifferenceHessian(unittest.TestCase):
    """Adaptive finite-difference Hessian used by analytic flows."""

    def test_non_quadratic_float64(self):
        """Matches the analytic Hessian of a non-quadratic function."""
        def function(x):
            return np.sin(x[0]) * np.exp(x[1]) + x[0]**2 * x[1]**3
        x = np.array([0.3, 0.5])
        expected = np.array([
            [-np.sin(0.3) * np.exp(0.5) + 2. * 0.5**3, np.cos(0.3) * np.exp(0.5) + 6. * 0.3 * 0.5**2],
            [np.cos(0.3) * np.exp(0.5) + 6. * 0.3 * 0.5**2, np.sin(0.3) * np.exp(0.5) + 6. * 0.3**2 * 0.5]])
        hessian = af._finite_difference_hessian(function, x)
        np.testing.assert_allclose(hessian, expected, atol=1e-8)
        np.testing.assert_allclose(hessian, hessian.T)

    def test_float32_density(self):
        """Stays accurate when the density is evaluated in float32."""
        inverse_cov = np.array([[1.5, -0.4], [-0.4, 3.0]], dtype=np.float32)

        def function(x):
            x = np.asarray(x, dtype=np.float32)
            return np.float32(-0.5) * x @ inverse_cov @ x + np.float32(1.3)
        hessian = af._finite_difference_hessian(function, np.array([0.7, -1.2]))
        np.testing.assert_allclose(hessian, -inverse_cov, atol=1e-5)

    def test_output_shapes(self):
        """Scalar outputs give (D, D) and size-1 outputs (1, D, D)."""
        x = np.array([0.1, 0.2, 0.3])
        self.assertEqual(af._finite_difference_hessian(lambda z: float(np.sum(z**2)), x).shape, (3, 3))
        self.assertEqual(af._finite_difference_hessian(lambda z: np.array([np.sum(z**2)]), x).shape, (1, 3, 3))
        np.testing.assert_allclose(af._finite_difference_hessian(lambda z: float(np.sum(z**2)), x), 2. * np.eye(3),
                                   atol=1e-8)

    def test_integer_valued_function(self):
        """Non floating outputs use the float64 round-off: a constant integer function has zero Hessian."""
        x = np.array([0.4, -0.7])
        hessian = af._finite_difference_hessian(lambda z: np.int64(3), x)
        self.assertEqual(hessian.shape, (2, 2))
        np.testing.assert_array_equal(hessian, np.zeros((2, 2)))

    def test_hessian_from_pdf_only_distribution(self):
        """Without ``log_pdf`` the Hessian is computed from ``log(pdf)``."""
        cov = np.array([[1.0, 0.4], [0.4, 0.8]])
        inverse_cov = np.linalg.inv(cov)

        class GaussianPDF(DummyNumpyDist):
            """Two dimensional Gaussian exposing only its density."""
            def __init__(self):
                """Init."""
                super().__init__()
                self.names = ["x", "y"]
                self.labels = ["x", "y"]

            def pdf(self, x):
                """Evaluate the Gaussian density.

                :param x: point with shape ``(2,)``.
                :returns: density value.
                """
                x = np.asarray(x)
                return np.exp(-0.5 * x @ inverse_cov @ x) / (2. * np.pi * np.sqrt(np.linalg.det(cov)))

        flow = af.analytic_flow(GaussianPDF())
        self.assertFalse(flow.has_log_pdf)
        hessian = tu.to_numpy(flow.log_probability_hessian(np.array([0.3, -0.2])))
        self.assertEqual(hessian.shape, (2, 2))
        tolerance = 1e-8 if tu.get_precision() == torch.float64 else 1e-4
        np.testing.assert_allclose(hessian, -inverse_cov, atol=tolerance)

    def test_analytic_flow_hessian_of_gaussians(self):
        """Hessians of float32 and float64 wrapped Gaussians equal minus the inverse covariance."""
        cov = np.array([[1.0, 0.3, 0.1], [0.3, 0.5, 0.2], [0.1, 0.2, 2.0]])
        expected = -np.linalg.inv(cov)
        coords = np.array([[0.2, -0.3, 0.5], [1.0, 0.4, -1.5]])
        for precision, tolerance in [(torch.float32, 1e-4), (torch.float64, 1e-8)]:
            dist = torch.distributions.MultivariateNormal(
                torch.zeros(3, dtype=precision), torch.as_tensor(cov, dtype=precision))
            flow = af.analytic_flow(af.torch_prob_wrapper(dist, prec=precision))
            single = tu.to_numpy(flow.log_probability_hessian(coords[0]))
            batch = tu.to_numpy(flow.log_probability_hessian(coords))
            self.assertEqual(single.shape, (3, 3))
            self.assertEqual(batch.shape, (2, 3, 3))
            np.testing.assert_allclose(single, expected, atol=tolerance)
            np.testing.assert_allclose(batch, np.broadcast_to(expected, (2, 3, 3)), atol=tolerance)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
