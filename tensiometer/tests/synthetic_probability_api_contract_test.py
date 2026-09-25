"""
Contract tests of the public flow API.

The flows are built on Gaussian chains without trainable or prior bijectors, so that the
flow is exactly the Gaussian approximation of the chain and every public method has a
closed form. A short real training run checks that the flows learn.
"""

#########################################################################################################
# Imports

import unittest

import matplotlib
matplotlib.use('agg')

import numpy as np
import scipy.stats
import torch
from getdist import MCSamples

import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Helpers


def _gaussian_chain(mean, cov, num_samples, seed):
    """
    Gaussian getdist chain with exact log posterior values.

    :param mean: mean vector.
    :param cov: covariance matrix.
    :param num_samples: number of samples.
    :param seed: random seed.
    :returns: :class:`getdist.MCSamples`.
    """
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean, cov, size=num_samples)
    loglikes = -scipy.stats.multivariate_normal(mean, cov).logpdf(samples)
    names = ['p' + str(i) for i in range(len(mean))]
    return MCSamples(samples=samples, loglikes=loglikes, names=names, labels=names)


def _tolerance(float32_value, float64_value):
    """Tolerance for the active precision."""
    if tu.get_precision() == torch.float64:
        return float64_value
    return float32_value

#########################################################################################################
# Gaussian flow contract


class TestGaussianFlowContract(unittest.TestCase):
    """Closed form checks on a flow that is exactly Gaussian."""

    @classmethod
    def setUpClass(cls):
        """Build the Gaussian flow once."""
        cls.true_mean = np.array([1., -1., 0.5])
        cls.true_cov = np.array([[1., 0.5, 0.2], [0.5, 2., 0.3], [0.2, 0.3, 0.5]])
        cls.chain = _gaussian_chain(cls.true_mean, cls.true_cov, 5000, seed=1)
        cls.flow = sp.FlowCallback(cls.chain, prior_bijector=None, trainable_bijector=None, feedback=0)
        affine = cls.flow.bijectors[1]
        cls.mean = tu.to_numpy(affine.shift).astype(np.float64)
        cls.scale_tril = tu.to_numpy(affine.scale_tril).astype(np.float64)
        cls.cov = cls.scale_tril @ cls.scale_tril.T
        cls.points = cls.chain.samples[:7]

    def test_gaussian_approximation(self):
        """The whitening bijector is the Gaussian approximation of the chain."""
        self.assertTrue(np.allclose(self.mean, self.true_mean, atol=0.1))
        self.assertTrue(np.allclose(self.cov, self.true_cov, atol=0.1))

    def test_log_probability(self):
        """Log probability matches the multivariate normal density."""
        expected = scipy.stats.multivariate_normal(self.mean, self.cov).logpdf(self.points)
        result = self.flow.log_probability(self.points)
        self.assertEqual(result.dtype, tu.get_precision())
        self.assertEqual(result.device.type, 'cpu')
        self.assertTrue(np.allclose(result.numpy(), expected, atol=_tolerance(1e-4, 1e-10)))

    def test_log_probability_derivatives(self):
        """Gradient and Hessian of the Gaussian log density."""
        inv_cov = np.linalg.inv(self.cov)
        expected_gradient = -(self.points - self.mean) @ inv_cov
        gradient = self.flow.log_probability_jacobian(self.points).numpy()
        hessian = self.flow.log_probability_hessian(self.points).numpy()
        self.assertTrue(np.allclose(gradient, expected_gradient, atol=_tolerance(1e-4, 1e-10)))
        self.assertTrue(np.allclose(hessian, -inv_cov[None], atol=_tolerance(1e-4, 1e-10)))

    def test_metric(self):
        """The metric is the inverse covariance, constant in space."""
        metric = self.flow.metric(self.points).numpy()
        inverse_metric = self.flow.inverse_metric(self.points).numpy()
        self.assertTrue(np.allclose(metric, np.linalg.inv(self.cov)[None], atol=_tolerance(1e-4, 1e-10)))
        self.assertTrue(np.allclose(inverse_metric, self.cov[None], atol=_tolerance(1e-4, 1e-10)))
        self.assertTrue(np.allclose(self.flow.coord_metric_derivative(self.points).numpy(), 0.))
        self.assertTrue(np.allclose(self.flow.coord_inverse_metric_derivative(self.points).numpy(), 0.))
        self.assertTrue(np.allclose(self.flow.levi_civita_connection(self.points).numpy(), 0.))

    def test_log_det_metric(self):
        """Log determinant of the metric is minus the log determinant of the covariance."""
        result = self.flow.log_det_metric(self.points).numpy()
        self.assertEqual(result.shape, (len(self.points),))
        self.assertTrue(np.allclose(result, -np.log(np.linalg.det(self.cov)), atol=_tolerance(1e-4, 1e-10)))

    def test_jacobians_are_inverse(self):
        """Direct and inverse Jacobians are inverse matrices."""
        product = self.flow.direct_jacobian(self.points) @ self.flow.inverse_jacobian(self.points)
        identity = np.broadcast_to(np.eye(3), product.shape)
        self.assertTrue(np.allclose(product.numpy(), identity, atol=_tolerance(1e-4, 1e-10)))

    def test_geodesic_distance(self):
        """Geodesic distance is the Mahalanobis distance."""
        inv_cov = np.linalg.inv(self.cov)
        diff = self.points[:3] - self.points[3:6]
        expected = np.sqrt(np.einsum('ni,ij,nj->n', diff, inv_cov, diff))
        result = self.flow.geodesic_distance(self.points[:3], self.points[3:6], axis=-1).numpy()
        self.assertTrue(np.allclose(result, expected, atol=_tolerance(1e-4, 1e-10)))

    def test_geodesic_bvp(self):
        """Geodesics of a Gaussian flow are straight lines."""
        trajectory = self.flow.geodesic_bvp(self.points[:2], self.points[2:4], num_points=5).numpy()
        self.assertEqual(trajectory.shape, (2, 5, 3))
        expected = self.points[:2, None, :] + np.linspace(0., 1., 5)[None, :, None] \
            * (self.points[2:4] - self.points[:2])[:, None, :]
        self.assertTrue(np.allclose(trajectory, expected, atol=_tolerance(1e-4, 1e-10)))

    def test_abstract_coordinates(self):
        """Abstract coordinates are the whitened points."""
        abstract = self.flow.map_to_abstract_coord(self.points).numpy()
        expected = np.linalg.solve(self.scale_tril, (self.points - self.mean).T).T
        self.assertTrue(np.allclose(abstract, expected, atol=_tolerance(1e-4, 1e-10)))
        back = self.flow.map_to_original_coord(abstract).numpy()
        self.assertTrue(np.allclose(back, self.points, atol=_tolerance(1e-4, 1e-10)))
        self.assertTrue(np.allclose(self.flow.log_probability_abs(abstract).numpy(),
                                    self.flow.log_probability(self.points).numpy(), atol=_tolerance(1e-4, 1e-10)))

    def test_samples(self):
        """Samples have the covariance of the flow."""
        torch.manual_seed(0)
        samples = self.flow.sample(20000)
        self.assertEqual(samples.shape, (20000, 3))
        self.assertEqual(samples.device.type, 'cpu')
        self.assertTrue(np.allclose(np.cov(samples.numpy().T), self.cov, atol=0.1))
        self.assertTrue(np.allclose(np.mean(samples.numpy(), axis=0), self.mean, atol=0.05))

    def test_public_outputs(self):
        """Public outputs are detached CPU tensors that support numpy conversion."""
        for method in [self.flow.log_probability, self.flow.log_probability_jacobian, self.flow.metric,
                       self.flow.map_to_abstract_coord, self.flow.log_det_metric]:
            result = method(self.points)
            self.assertTrue(torch.is_tensor(result))
            self.assertFalse(result.requires_grad)
            self.assertEqual(result.device.type, 'cpu')
            self.assertIsInstance(np.asarray(result), np.ndarray)
        # tensors, lists and numpy arrays are accepted:
        reference = self.flow.log_probability(self.points).numpy()
        for value in [torch.as_tensor(self.points), self.points.tolist(), self.points.astype(np.float32)]:
            self.assertTrue(np.allclose(self.flow.log_probability(value).numpy(), reference, atol=_tolerance(1e-4, 1e-10)))
        # comparison as used by mcmc_tension:
        threshold = self.flow.log_probability(self.flow.cast([self.mean]))[0]
        self.assertEqual(np.asarray(self.flow.log_probability(self.points) > threshold).dtype, np.bool_)

    def test_graph_attached_outputs(self):
        """Inputs that require gradients give outputs attached to the graph."""
        x = torch.as_tensor(self.points, dtype=tu.get_precision()).requires_grad_(True)
        log_prob = self.flow.log_probability(x)
        self.assertTrue(log_prob.requires_grad)
        (gradient,) = torch.autograd.grad(log_prob.sum(), x)
        self.assertTrue(np.allclose(gradient.numpy(), self.flow.log_probability_jacobian(self.points).numpy(),
                                    atol=_tolerance(1e-4, 1e-10)))
        jacobian = self.flow.log_probability_jacobian(x)
        self.assertTrue(jacobian.requires_grad)
        (hessian_row,) = torch.autograd.grad(jacobian[:, 0].sum(), x)
        self.assertTrue(np.allclose(hessian_row.numpy(), self.flow.log_probability_hessian(self.points).numpy()[:, 0, :],
                                    atol=_tolerance(1e-4, 1e-10)))

    def test_mcsamples(self):
        """MCSamples with and without log likelihoods."""
        torch.manual_seed(1)
        samples = self.flow.MCSamples(2000)
        self.assertTrue(np.allclose(samples.getMeans(), self.mean, atol=0.1))
        self.assertIsNotNone(samples.loglikes)
        samples = self.flow.MCSamples(100, logLikes=False)
        self.assertIsNone(samples.loglikes)

#########################################################################################################
# Training contract


class TestTrainingContract(unittest.TestCase):
    """A short real training run improves the flow."""

    def test_training_improves_validation_loss(self):
        """Validation loss decreases for a correlated non-Gaussian target."""
        np.random.seed(0)
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        x = rng.normal(size=3000)
        y = 0.5 * x**2 + 0.3 * rng.normal(size=3000)
        chain = MCSamples(samples=np.stack([x, y], axis=1), names=['x', 'y'], labels=['x', 'y'])
        flow = sp.FlowCallback(chain, prior_bijector=None, feedback=0, plot_every=0,
                               hidden_units=[16, 16], n_transformations=2, permutations=False)
        history = flow.train(epochs=20, verbose=0, callbacks=[])
        val_loss = history.history['val_loss']
        self.assertEqual(len(val_loss), 20)
        # the Gaussian approximation has loss log(2 pi e) ~ 2.84, the target about 1.9:
        self.assertLess(np.mean(val_loss[-3:]), val_loss[0] - 0.2)
        self.assertLess(np.mean(val_loss[-3:]), 2.3)
        self.assertTrue(flow.is_trained)
        self.assertEqual(len(flow.log['chi2Z_ks_p']), 20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
