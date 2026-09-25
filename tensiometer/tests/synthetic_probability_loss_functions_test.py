"""Tests for the synthetic probability loss functions (PyTorch implementation)."""

#########################################################################################################
# Imports

import unittest
from unittest.mock import patch

import numpy as np
import torch

from tensiometer.synthetic_probability import loss_functions as lf
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Helpers


def _tensor(values):
    """Tensor with the active precision on the default device."""
    return tu.to_tensor(values)


def _to_float(value):
    """Python float of a scalar tensor."""
    return float(tu.to_numpy(value))


def _tolerance():
    """Relative tolerance matching the active precision."""
    if tu.get_precision() == torch.float64:
        return 1e-12
    return 1e-6

#########################################################################################################
# Reduction helpers


class TestReductionHelpers(unittest.TestCase):
    """Tests of ``_reduce_weighted_loss`` and ``_broadcast_sample_weight``."""

    def setUp(self):
        """Set up per-sample losses and weights."""
        self.losses = _tensor([1.0, 2.0, 3.0])
        self.weights = _tensor([1.0, 1.0, 2.0])

    def test_reduction_none(self):
        """'none' returns the weighted per-sample losses."""
        out = lf._reduce_weighted_loss(self.losses, self.weights, 'none')
        np.testing.assert_allclose(tu.to_numpy(out), [1.0, 2.0, 6.0])
        out = lf._reduce_weighted_loss(self.losses, None, 'none')
        np.testing.assert_allclose(tu.to_numpy(out), [1.0, 2.0, 3.0])

    def test_reduction_sum(self):
        """'sum' returns the sum of the weighted losses."""
        out = lf._reduce_weighted_loss(self.losses, self.weights, 'sum')
        self.assertAlmostEqual(_to_float(out), 9.0, places=5)
        out = lf._reduce_weighted_loss(self.losses, None, 'sum')
        self.assertAlmostEqual(_to_float(out), 6.0, places=5)

    def test_reduction_weighted_mean(self):
        """'weighted_mean' divides by the sum of the weights."""
        out = lf._reduce_weighted_loss(self.losses, self.weights, 'weighted_mean')
        self.assertAlmostEqual(_to_float(out), 9.0 / 4.0, places=5)
        out = lf._reduce_weighted_loss(self.losses, None, 'weighted_mean')
        self.assertAlmostEqual(_to_float(out), 2.0, places=5)

    def test_reduction_weighted_mean_zero_weights(self):
        """A zero weight sum is replaced by one."""
        out = lf._reduce_weighted_loss(self.losses, _tensor([0.0, 0.0, 0.0]), 'weighted_mean')
        self.assertEqual(_to_float(out), 0.0)
        self.assertTrue(np.isfinite(_to_float(out)))

    def test_reduction_sum_over_batch_size(self):
        """'sum_over_batch_size' divides by the number of losses (Keras default)."""
        out = lf._reduce_weighted_loss(self.losses, self.weights, 'sum_over_batch_size')
        self.assertAlmostEqual(_to_float(out), 3.0, places=5)
        out = lf._reduce_weighted_loss(self.losses, None, 'sum_over_batch_size')
        self.assertAlmostEqual(_to_float(out), 2.0, places=5)

    def test_reduction_keeps_dtype(self):
        """Reductions keep the dtype of the losses."""
        for reduction in ['none', 'sum', 'weighted_mean', 'sum_over_batch_size']:
            out = lf._reduce_weighted_loss(self.losses, self.weights, reduction)
            self.assertEqual(out.dtype, tu.get_precision())

    def test_invalid_reduction(self):
        """Unknown reductions raise ValueError."""
        with self.assertRaises(ValueError):
            lf._reduce_weighted_loss(self.losses, None, 'mean')
        with self.assertRaises(ValueError):
            lf.Loss(reduction='bad')
        with self.assertRaises(ValueError):
            lf.standard_loss(reduction='auto')

    def test_broadcast_none(self):
        """No weights stay None."""
        self.assertIsNone(lf._broadcast_sample_weight(None, self.losses))

    def test_broadcast_scalar(self):
        """A scalar weight is broadcast to the loss shape."""
        out = lf._broadcast_sample_weight(2.0, self.losses)
        self.assertEqual(tuple(out.shape), (3,))
        np.testing.assert_allclose(tu.to_numpy(out), [2.0, 2.0, 2.0])

    def test_broadcast_trailing_axes(self):
        """Weights ``(N,)`` are reshaped to ``(N, 1)`` and broadcast along the trailing axis."""
        losses = _tensor([[1.0, 2.0], [3.0, 4.0]])
        out = lf._broadcast_sample_weight([1.0, 2.0], losses)
        self.assertEqual(tuple(out.shape), (2, 2))
        np.testing.assert_allclose(tu.to_numpy(out), [[1.0, 1.0], [2.0, 2.0]])

    def test_broadcast_dtype_and_device(self):
        """Broadcast weights follow the dtype and device of the losses."""
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = lf._broadcast_sample_weight(weights, self.losses)
        self.assertEqual(out.dtype, self.losses.dtype)
        self.assertEqual(out.device, self.losses.device)

#########################################################################################################
# Base classes


class TestBaseLoss(unittest.TestCase):
    """Tests of the ``Loss`` base class and ``mean_squared_error``."""

    def test_base_loss_not_implemented(self):
        """The base class does not implement the per-sample loss."""
        loss = lf.Loss()
        self.assertEqual(loss.reduction, 'weighted_mean')
        with self.assertRaises(NotImplementedError):
            loss(None, _tensor([1.0, 2.0]))
        self.assertIsNone(loss.reset())

    def test_mean_squared_error(self):
        """Mean squared error over the last axis, then weighted mean over samples."""
        y_true = _tensor([[1.0, 2.0], [0.0, 0.0]])
        y_pred = _tensor([[1.0, 0.0], [1.0, 1.0]])
        loss = lf.mean_squared_error()
        per_sample = loss.compute_loss(y_true, y_pred, None)
        np.testing.assert_allclose(tu.to_numpy(per_sample), [2.0, 1.0], rtol=_tolerance())
        self.assertAlmostEqual(_to_float(loss(y_true, y_pred)), 1.5, places=5)
        weighted = loss(y_true, y_pred, sample_weight=_tensor([3.0, 1.0]))
        self.assertAlmostEqual(_to_float(weighted), (3.0 * 2.0 + 1.0) / 4.0, places=5)

    def test_mean_squared_error_gradient(self):
        """The mean squared error is differentiable with respect to the predictions."""
        y_true = _tensor([[1.0, 2.0]])
        y_pred = _tensor([[0.0, 0.0]]).requires_grad_(True)
        value = lf.mean_squared_error()(y_true, y_pred)
        value.backward()
        np.testing.assert_allclose(tu.to_numpy(y_pred.grad), [[-1.0, -2.0]], rtol=_tolerance())

#########################################################################################################
# Density losses


class TestDensityLosses(unittest.TestCase):
    """Tests of the standard and combined density / evidence-error losses."""

    def setUp(self):
        """Set up log densities and weights."""
        self.y_true = _tensor([0.5, 1.0, -0.2])
        self.y_pred = _tensor([0.4, 0.8, 0.1])
        self.sample_weight = _tensor([1.0, 2.0, 1.0])
        self.np_true = np.array([0.5, 1.0, -0.2])
        self.np_pred = np.array([0.4, 0.8, 0.1])
        self.np_weight = np.array([1.0, 2.0, 1.0])

    def _expected_components(self, beta):
        """Numpy density loss and evidence-error loss."""
        diffs = self.np_true - self.np_pred
        mean_diff = np.sum(diffs * self.np_weight) / np.sum(self.np_weight)
        return -(self.np_pred + beta), (diffs - mean_diff)**2

    def test_standard_loss(self):
        """The standard loss is the weighted mean of ``-y_pred``."""
        loss = lf.standard_loss()
        value = loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        expected = np.average(-self.np_pred, weights=self.np_weight)
        self.assertAlmostEqual(_to_float(value), expected, places=5)
        self.assertEqual(value.dtype, tu.get_precision())
        self.assertEqual(value.dim(), 0)

    def test_standard_loss_without_targets_and_weights(self):
        """The standard loss accepts ``y_true=None`` and no weights."""
        loss = lf.standard_loss()
        value = loss(None, self.y_pred)
        self.assertAlmostEqual(_to_float(value), -np.mean(self.np_pred), places=5)

    def test_standard_loss_helpers(self):
        """Components, ``call``, feedback and reset of the standard loss."""
        loss = lf.standard_loss(reduction='sum')
        self.assertAlmostEqual(_to_float(loss(None, self.y_pred)), -np.sum(self.np_pred), places=5)
        components = loss.compute_loss_components(self.y_true, self.y_pred, self.sample_weight)
        np.testing.assert_allclose(tu.to_numpy(components), -self.np_pred, rtol=_tolerance())
        np.testing.assert_allclose(tu.to_numpy(loss.call(self.y_true, self.y_pred)), -self.np_pred, rtol=_tolerance())
        with patch('builtins.print') as mock_print:
            loss.print_feedback(padding='  ')
        self.assertIn('standard', mock_print.call_args[0][0])
        self.assertIsNone(loss.reset())

    def test_standard_loss_gradient(self):
        """Gradients of the standard loss are ``-w / sum(w)``."""
        y_pred = self.y_pred.clone().requires_grad_(True)
        lf.standard_loss()(None, y_pred, sample_weight=self.sample_weight).backward()
        expected = -self.np_weight / np.sum(self.np_weight)
        np.testing.assert_allclose(tu.to_numpy(y_pred.grad), expected, rtol=_tolerance())

    def test_constant_weight_loss(self):
        """The constant weight loss combines the two components with ``alpha``."""
        alpha, beta = 0.7, 0.1
        loss = lf.constant_weight_loss(alpha=alpha, beta=beta)
        value = loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        loss_1, loss_2 = self._expected_components(beta)
        expected = np.average(alpha * loss_1 + (1. - alpha) * loss_2, weights=self.np_weight)
        self.assertAlmostEqual(_to_float(value), expected, places=5)
        components = loss.compute_loss_components(self.y_true, self.y_pred, self.sample_weight)
        self.assertEqual(len(components), 2)
        np.testing.assert_allclose(tu.to_numpy(components[0]), loss_1, rtol=1e-5)
        np.testing.assert_allclose(tu.to_numpy(components[1]), loss_2, rtol=1e-4, atol=1e-7)
        with patch('builtins.print') as mock_print:
            loss.print_feedback(padding='')
        self.assertEqual(mock_print.call_count, 2)
        loss.reset()
        self.assertEqual(loss.alpha, alpha)
        self.assertEqual(loss.beta, beta)

    def test_constant_weight_loss_default_weights(self):
        """Without weights all samples count the same."""
        loss = lf.constant_weight_loss(alpha=0.5, beta=0.0)
        value = loss(self.y_true, self.y_pred)
        self.np_weight = np.ones(3)
        loss_1, loss_2 = self._expected_components(0.0)
        self.assertAlmostEqual(_to_float(value), np.mean(0.5 * loss_1 + 0.5 * loss_2), places=5)

    def test_constant_weight_loss_components_without_weights(self):
        """Loss components computed with ``sample_weight=None`` use unit weights."""
        loss = lf.constant_weight_loss(alpha=0.5, beta=0.2)
        components = loss.compute_loss_components(self.y_true, self.y_pred, None)
        self.np_weight = np.ones(3)
        loss_1, loss_2 = self._expected_components(0.2)
        np.testing.assert_allclose(tu.to_numpy(components[0]), loss_1, rtol=1e-5)
        np.testing.assert_allclose(tu.to_numpy(components[1]), loss_2, rtol=1e-4, atol=1e-7)
        weighted = loss.compute_loss_components(self.y_true, self.y_pred, _tensor([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(tu.to_numpy(components[1]), tu.to_numpy(weighted[1]), rtol=_tolerance())

    def test_constant_weight_loss_numpy_targets(self):
        """Targets given as numpy arrays are converted to the prediction dtype."""
        loss = lf.constant_weight_loss(alpha=0.0)
        value = loss(self.np_true, self.y_pred, sample_weight=self.sample_weight)
        self.assertEqual(value.dtype, tu.get_precision())
        self.assertTrue(np.isfinite(_to_float(value)))

#########################################################################################################
# Variable weight losses


class TestVariableWeightLosses(unittest.TestCase):
    """Tests of the variable weight losses and of their ``reset``."""

    def setUp(self):
        """Set up log densities and weights."""
        self.y_true = _tensor([0.5, 1.0, -0.2])
        self.y_pred = _tensor([0.4, 0.8, 0.1])
        self.sample_weight = _tensor([1.0, 2.0, 1.0])

    def _assert_float_lambdas(self, loss):
        """Lambdas and beta are python floats."""
        self.assertIsInstance(loss.lambda_1, float)
        self.assertIsInstance(loss.lambda_2, float)
        self.assertIsInstance(loss.beta, float)

    def test_variable_weight_loss_base(self):
        """Base class: value, components, abstract update and feedback."""
        loss = lf.variable_weight_loss(lambda_1=0.25, lambda_2=0.75, beta=0.1)
        self._assert_float_lambdas(loss)
        value = loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        loss_1, loss_2, lambda_1, lambda_2 = loss.compute_loss_components(
            self.y_true, self.y_pred, self.sample_weight)
        self.assertEqual((lambda_1, lambda_2), (0.25, 0.75))
        expected = np.average(0.25 * tu.to_numpy(loss_1) + 0.75 * tu.to_numpy(loss_2), weights=[1.0, 2.0, 1.0])
        self.assertAlmostEqual(_to_float(value), expected, places=5)
        components = loss.compute_loss_components(self.y_true, self.y_pred, self.sample_weight, lambda_1=0.5, lambda_2=0.1)
        self.assertEqual(len(components), 4)
        self.assertEqual(components[2:], (0.5, 0.1))
        with self.assertRaises(NotImplementedError):
            loss.update_lambda_values_on_epoch_begin(1)
        with self.assertRaises(NotImplementedError):
            loss.print_feedback(padding='')

    def test_variable_weight_loss_default_weights(self):
        """The variable weight loss works without weights."""
        loss = lf.variable_weight_loss(lambda_1=0.5, lambda_2=0.5)
        value = loss(self.y_true, self.y_pred)
        self.assertTrue(np.isfinite(_to_float(value)))

    def test_variable_weight_loss_reset(self):
        """Reset restores the initial lambdas and beta."""
        loss = lf.variable_weight_loss(lambda_1=0.25, lambda_2=0.75, beta=0.1)
        loss.lambda_1, loss.lambda_2, loss.beta = 0.9, 0.1, 3.0
        loss.reset()
        self.assertEqual((loss.lambda_1, loss.lambda_2, loss.beta), (0.25, 0.75, 0.1))
        self._assert_float_lambdas(loss)

    def test_random_weight_loss(self):
        """Random weights are drawn only after ``initial_random_epoch``."""
        np.random.seed(0)
        loss = lf.random_weight_loss(initial_random_epoch=2, lambda_1=0.5, beta=0.0)
        loss.update_lambda_values_on_epoch_begin(1)
        self.assertEqual((loss.lambda_1, loss.lambda_2), (0.5, 0.5))
        values = set()
        for epoch in range(3, 30):
            loss.update_lambda_values_on_epoch_begin(epoch)
            self._assert_float_lambdas(loss)
            self.assertIn(loss.lambda_1, [0.0, 1.0])
            self.assertEqual(loss.lambda_2, 1.0 - loss.lambda_1)
            values.add(loss.lambda_1)
        self.assertEqual(values, {0.0, 1.0})
        with patch('builtins.print') as mock_print:
            loss.print_feedback(padding='')
        self.assertIn('randomized', mock_print.call_args[0][0])

    def test_random_weight_loss_reset(self):
        """Reset restores the lambdas and keeps ``initial_random_epoch``."""
        np.random.seed(1)
        loss = lf.random_weight_loss(initial_random_epoch=4, lambda_1=0.3)
        for epoch in range(5, 15):
            loss.update_lambda_values_on_epoch_begin(epoch)
        loss.reset()
        self.assertEqual(loss.initial_random_epoch, 4)
        self.assertAlmostEqual(loss.lambda_1, 0.3)
        self.assertAlmostEqual(loss.lambda_2, 0.7)

    def test_annealed_weight_loss(self):
        """Annealing multiplies lambda_1 by ``exp(-(epoch - anneal_epoch) / roll_off_nepoch)``."""
        loss = lf.annealed_weight_loss(anneal_epoch=0, lambda_1=1.0, beta=0.0, roll_off_nepoch=2)
        loss.update_lambda_values_on_epoch_begin(1)
        self._assert_float_lambdas(loss)
        self.assertAlmostEqual(loss.lambda_1, np.exp(-0.5))
        self.assertAlmostEqual(loss.lambda_2, 1.0 - np.exp(-0.5))
        loss.update_lambda_values_on_epoch_begin(2)
        self.assertAlmostEqual(loss.lambda_1, np.exp(-0.5) * np.exp(-1.0))
        with patch('builtins.print') as mock_print:
            loss.print_feedback(padding='')
        self.assertIn('annealed', mock_print.call_args[0][0])

    def test_annealed_weight_loss_noop(self):
        """Before ``anneal_epoch`` the lambdas are unchanged."""
        loss = lf.annealed_weight_loss(anneal_epoch=5, lambda_1=0.8, beta=0.0, roll_off_nepoch=1)
        loss.update_lambda_values_on_epoch_begin(2)
        self.assertAlmostEqual(loss.lambda_1, 0.8)
        self.assertAlmostEqual(loss.lambda_2, 0.2)

    def test_annealed_weight_loss_reset_keeps_hyperparameters(self):
        """Reset restores the lambdas and keeps ``anneal_epoch`` and ``roll_off_nepoch``."""
        loss = lf.annealed_weight_loss(anneal_epoch=3, lambda_1=0.9, beta=0.2, roll_off_nepoch=7)
        for epoch in range(10):
            loss.update_lambda_values_on_epoch_begin(epoch)
        self.assertLess(loss.lambda_1, 0.9)
        loss.reset()
        self.assertEqual(loss.anneal_epoch, 3)
        self.assertEqual(loss.roll_off_nepoch, 7)
        self.assertAlmostEqual(loss.lambda_1, 0.9)
        self.assertAlmostEqual(loss.lambda_2, 0.1)
        self.assertAlmostEqual(loss.beta, 0.2)
        self._assert_float_lambdas(loss)

    def test_softadapt_zero_rate(self):
        """With fewer than two log entries the density weight is one."""
        loss = lf.SoftAdapt_weight_loss(tau=1.0, smoothing=False)
        logs = {'val_rho_loss': [1.0], 'val_ee_loss': [1.0]}
        loss.update_lambda_values_on_epoch_begin(0, logs=logs)
        self.assertEqual((loss.lambda_1, loss.lambda_2), (1.0, 0.0))
        self._assert_float_lambdas(loss)

    def test_softadapt_no_smoothing(self):
        """Without smoothing the lambdas are the softmax of the loss rates."""
        tau = 2.0
        loss = lf.SoftAdapt_weight_loss(tau=tau, smoothing=False)
        logs = {'val_rho_loss': [1.0, 0.5], 'val_ee_loss': [1.0, 1.5]}
        loss.update_lambda_values_on_epoch_begin(2, logs=logs)
        expected = np.exp(tau * -0.5) / (np.exp(tau * -0.5) + np.exp(tau * 0.5))
        self.assertAlmostEqual(loss.lambda_1, expected)
        self.assertAlmostEqual(loss.lambda_2, 1.0 - expected)
        self._assert_float_lambdas(loss)

    def test_softadapt_smoothing(self):
        """Smoothing updates the rate and lambda buffers."""
        loss = lf.SoftAdapt_weight_loss(tau=1.0, smoothing=True, smoothing_tau=2)
        logs = {'val_rho_loss': [1.0, 0.9], 'val_ee_loss': [1.0, 1.1]}
        loss.update_lambda_values_on_epoch_begin(1, logs=logs)
        self.assertAlmostEqual(loss.rate_1_buffer, 0.5 * -0.1)
        self.assertAlmostEqual(loss.rate_2_buffer, 0.5 * 0.1)
        self.assertNotEqual(loss.lambda_1_buffer, 1.0)
        self.assertGreater(loss.lambda_1, 0.0)
        self.assertLess(loss.lambda_1, 1.0)
        self.assertAlmostEqual(loss.lambda_1 + loss.lambda_2, 1.0)
        self._assert_float_lambdas(loss)
        with patch('builtins.print') as mock_print:
            loss.print_feedback(padding='')
        self.assertEqual(mock_print.call_count, 4)

    def test_softadapt_custom_quantities(self):
        """The monitored log keys can be changed."""
        loss = lf.SoftAdapt_weight_loss(smoothing=False, quantity_1='a', quantity_2='b')
        loss.update_lambda_values_on_epoch_begin(1, logs={'a': [2.0, 1.0], 'b': [1.0, 1.0]})
        self.assertAlmostEqual(loss.lambda_1, np.exp(-1.0) / (np.exp(-1.0) + 1.0))

    def test_softadapt_missing_logs_raise(self):
        """Missing logs or monitored quantities raise a clear ValueError."""
        loss = lf.SoftAdapt_weight_loss()
        with self.assertRaises(ValueError):
            loss.update_lambda_values_on_epoch_begin(1)
        with self.assertRaises(ValueError):
            loss.update_lambda_values_on_epoch_begin(1, logs={'val_rho_loss': [1.0]})

    def test_softadapt_smoothing_buffers_normalized(self):
        """Smoothed lambda buffers are normalized by the same total and sum to one."""
        loss = lf.SoftAdapt_weight_loss(tau=1.0, smoothing=True, smoothing_tau=2)
        logs = {'val_rho_loss': [1.0, 0.5], 'val_ee_loss': [1.0, 1.5]}
        loss.update_lambda_values_on_epoch_begin(1, logs=logs)
        self.assertAlmostEqual(loss.lambda_1_buffer + loss.lambda_2_buffer, 1.0)
        self.assertAlmostEqual(loss.lambda_1, loss.lambda_1_buffer)
        self.assertAlmostEqual(loss.lambda_2, loss.lambda_2_buffer)

    def test_softadapt_reset_clears_buffers(self):
        """Reset clears the SoftAdapt buffers and keeps the hyperparameters."""
        loss = lf.SoftAdapt_weight_loss(tau=3.0, beta=0.5, smoothing=True, smoothing_tau=4,
                                        quantity_1='a', quantity_2='b')
        loss.update_lambda_values_on_epoch_begin(1, logs={'a': [2.0, 1.0], 'b': [1.0, 1.5]})
        self.assertNotEqual(loss.rate_1_buffer, 0.0)
        loss.reset()
        self.assertEqual(loss.rate_1_buffer, 0.0)
        self.assertEqual(loss.rate_2_buffer, 0.0)
        self.assertEqual(loss.lambda_1_buffer, 1.0)
        self.assertEqual(loss.lambda_2_buffer, 0.0)
        self.assertEqual((loss.lambda_1, loss.lambda_2, loss.beta), (1.0, 0.0, 0.5))
        self.assertEqual(loss.tau, 3.0)
        self.assertEqual(loss.smoothing_alpha, 0.25)
        self.assertTrue(loss.smoothing)
        self.assertEqual((loss.quantity_1, loss.quantity_2), ('a', 'b'))

    def test_sharpstep(self):
        """SharpStep switches the density weight at ``step_epoch``."""
        loss = lf.SharpStep(step_epoch=2, value_1=1.0, value_2=0.25, beta=0.0)
        loss.update_lambda_values_on_epoch_begin(1)
        self.assertEqual((loss.lambda_1, loss.lambda_2), (1.0, 0.0))
        loss.update_lambda_values_on_epoch_begin(2)
        self.assertEqual((loss.lambda_1, loss.lambda_2), (0.25, 0.75))
        self._assert_float_lambdas(loss)
        with patch('builtins.print') as mock_print:
            loss.print_feedback(padding='')
        self.assertIn('sharp step', mock_print.call_args[0][0])

    def test_sharpstep_reset_keeps_hyperparameters(self):
        """Reset restores the lambdas and keeps ``step_epoch`` and the two values."""
        loss = lf.SharpStep(step_epoch=7, value_1=0.9, value_2=0.3, beta=0.4)
        loss.update_lambda_values_on_epoch_begin(10)
        self.assertEqual(loss.lambda_1, 0.3)
        loss.reset()
        self.assertEqual(loss.step_epoch, 7)
        self.assertEqual((loss.value_1, loss.value_2), (0.9, 0.3))
        self.assertEqual((loss.lambda_1, loss.lambda_2, loss.beta), (1.0, 0.0, 0.4))

    def test_variable_losses_follow_lambdas(self):
        """The value of a variable loss follows its current lambdas."""
        loss = lf.SharpStep(step_epoch=1, value_1=1.0, value_2=0.0)
        loss.update_lambda_values_on_epoch_begin(0)
        density_only = _to_float(loss(self.y_true, self.y_pred, sample_weight=self.sample_weight))
        loss.update_lambda_values_on_epoch_begin(1)
        evidence_only = _to_float(loss(self.y_true, self.y_pred, sample_weight=self.sample_weight))
        expected_density = _to_float(lf.standard_loss()(None, self.y_pred, sample_weight=self.sample_weight))
        self.assertAlmostEqual(density_only, expected_density, places=5)
        self.assertNotAlmostEqual(evidence_only, density_only, places=3)
        self.assertGreaterEqual(evidence_only, 0.0)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
