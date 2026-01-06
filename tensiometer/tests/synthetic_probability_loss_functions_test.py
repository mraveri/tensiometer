"""Tests for synthetic probability loss functions."""

#########################################################################################################
# Imports

import unittest

import numpy as np
import tensorflow as tf

from tensiometer.synthetic_probability import loss_functions as lf

#########################################################################################################
# Test cases


class TestLossFunctions(unittest.TestCase):
    """Loss functions test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.y_true = tf.constant([[0.5], [1.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[0.4], [0.8]], dtype=tf.float32)
        self.sample_weight = tf.constant([1.0, 2.0], dtype=tf.float32)

    def test_standard_loss(self):
        """Test standard loss."""
        loss = lf.standard_loss()
        val = loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        self.assertTrue(np.all(np.isfinite(val.numpy())))
        components = loss.compute_loss_components(self.y_true, self.y_pred, self.sample_weight)
        self.assertTrue(np.all(np.isfinite(components.numpy())))
        loss.print_feedback(padding='')
        loss.reset()

    def test_constant_weight_loss(self):
        """Test constant weight loss."""
        loss = lf.constant_weight_loss(alpha=0.7, beta=0.1)
        val = loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        self.assertTrue(np.all(np.isfinite(val.numpy())))
        comp = loss.compute_loss_components(self.y_true, self.y_pred, self.sample_weight)
        self.assertEqual(len(comp), 2)
        loss.print_feedback(padding='')
        loss.reset()

    def test_constant_weight_loss_default_weights(self):
        """Test constant weight loss with default weights."""
        loss = lf.constant_weight_loss(alpha=0.7, beta=0.1)
        val = loss(self.y_true, self.y_pred)
        self.assertTrue(np.all(np.isfinite(val.numpy())))

    def test_constant_weight_loss_graph_branch(self):
        """Test constant weight loss graph branch."""
        loss = lf.constant_weight_loss(alpha=0.5, beta=0.0)
        @tf.function
        def _compute():
            """Wrap the loss call in a graph context."""
            return loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        val = _compute()
        self.assertTrue(np.all(np.isfinite(val.numpy())))

    def test_variable_weight_loss(self):
        """Test variable weight loss."""
        loss = lf.variable_weight_loss(lambda_1=0.5, lambda_2=0.5, beta=0.0)
        val = loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        self.assertTrue(np.all(np.isfinite(val.numpy())))
        comp = loss.compute_loss_components(self.y_true, self.y_pred, self.sample_weight, lambda_1=0.5, lambda_2=0.5)
        self.assertEqual(len(comp), 4)
        with self.assertRaises(NotImplementedError):
            loss.update_lambda_values_on_epoch_begin(epoch=1)
        with self.assertRaises(NotImplementedError):
            loss.print_feedback(padding='')
        loss.reset()

    def test_variable_weight_loss_default_weights(self):
        """Test variable weight loss with default weights."""
        loss = lf.variable_weight_loss(lambda_1=0.5, lambda_2=0.5, beta=0.0)
        val = loss(self.y_true, self.y_pred)
        self.assertTrue(np.all(np.isfinite(val.numpy())))

    def test_variable_weight_loss_defaults_and_graph(self):
        """Test variable weight loss defaults and graph branch."""
        loss = lf.variable_weight_loss(lambda_1=0.25, lambda_2=0.75, beta=0.0)
        comp = loss.compute_loss_components(self.y_true, self.y_pred, self.sample_weight)
        self.assertEqual(len(comp), 4)
        @tf.function
        def _compute():
            """Wrap the loss call in a graph context."""
            return loss(self.y_true, self.y_pred, sample_weight=self.sample_weight)
        val = _compute()
        self.assertTrue(np.all(np.isfinite(val.numpy())))

    def test_random_weight_loss(self):
        """Test random weight loss."""
        loss = lf.random_weight_loss(initial_random_epoch=0, lambda_1=0.5, beta=0.0)
        loss.update_lambda_values_on_epoch_begin(epoch=0)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertIn(lam1, [0.0, 1.0, 0.5])
        loss.update_lambda_values_on_epoch_begin(epoch=2)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertIn(lam1, [0.0, 1.0])
        loss.print_feedback(padding='')

    def test_annealed_weight_loss(self):
        """Test annealed weight loss."""
        loss = lf.annealed_weight_loss(anneal_epoch=0, lambda_1=1.0, beta=0.0, roll_off_nepoch=1)
        loss.update_lambda_values_on_epoch_begin(epoch=1)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertLess(lam1, 1.0)
        loss.print_feedback(padding='')

    def test_annealed_weight_loss_noop(self):
        """Test annealed weight loss no-op branch."""
        loss = lf.annealed_weight_loss(anneal_epoch=5, lambda_1=0.8, beta=0.0, roll_off_nepoch=1)
        loss.update_lambda_values_on_epoch_begin(epoch=2)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertAlmostEqual(lam1, 0.8)

    def test_softadapt_loss(self):
        """Test SoftAdapt loss."""
        loss = lf.SoftAdapt_weight_loss(tau=1.0, smoothing=True, smoothing_tau=2)
        logs = {'val_rho_loss': [1.0, 0.9], 'val_ee_loss': [1.0, 1.1]}
        loss.update_lambda_values_on_epoch_begin(epoch=1, logs=logs)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertGreaterEqual(lam1, 0.0)
        loss.print_feedback(padding='')

    def test_softadapt_zero_rate_and_no_smoothing(self):
        """Test SoftAdapt zero rate with no smoothing."""
        loss = lf.SoftAdapt_weight_loss(tau=1.0, smoothing=False)
        logs = {'val_rho_loss': [1.0], 'val_ee_loss': [1.0]}
        loss.update_lambda_values_on_epoch_begin(epoch=0, logs=logs)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertEqual(lam1, 1.0)
        loss.print_feedback(padding='')

    def test_softadapt_no_smoothing_nonzero_rate(self):
        """Test SoftAdapt no smoothing with nonzero rate."""
        loss = lf.SoftAdapt_weight_loss(tau=1.0, smoothing=False)
        logs = {'val_rho_loss': [1.0, 0.5], 'val_ee_loss': [1.0, 1.5]}
        loss.update_lambda_values_on_epoch_begin(epoch=2, logs=logs)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertGreaterEqual(lam1, 0.0)
        loss.print_feedback(padding='')

    def test_sharpstep_loss(self):
        """Test SharpStep loss."""
        loss = lf.SharpStep(step_epoch=1, value_1=1.0, value_2=0.25, beta=0.0)
        loss.update_lambda_values_on_epoch_begin(epoch=0)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertEqual(lam1, 1.0)
        loss.update_lambda_values_on_epoch_begin(epoch=2)
        lam1 = tf.keras.backend.get_value(loss.lambda_1)
        self.assertEqual(lam1, 0.25)
        loss.print_feedback(padding='')

    def test_broadcast_sample_weight_helpers(self):
        """Test broadcast sample weight helper branches."""
        losses = tf.constant([1.0, 2.0], dtype=tf.float32)
        self.assertIsNone(lf._broadcast_sample_weight(None, losses))
        scalar_weight = tf.constant(2.0, dtype=tf.float32)
        out = lf._broadcast_sample_weight(scalar_weight, losses)
        self.assertTrue(np.allclose(out.numpy(), np.array([2.0, 2.0], dtype=np.float32)))

        @tf.function(input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ])
        def _wrap(weight, tensor):
            """Wrap broadcast helper in graph mode."""
            return lf._broadcast_sample_weight(weight, tensor)

        weight = tf.constant([1.0, 2.0], dtype=tf.float32)
        loss_grid = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        out = _wrap(weight, loss_grid)
        self.assertEqual(out.shape, loss_grid.shape)

    def test_broadcast_sample_weight_unknown_rank(self):
        """Test broadcast helper with unknown ranks."""
        losses = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        weights = tf.constant([1.0, 2.0], dtype=tf.float32)
        losses.set_shape(tf.TensorShape(None))
        weights.set_shape(tf.TensorShape(None))
        out = lf._broadcast_sample_weight(weights, losses)
        expected = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        self.assertTrue(np.allclose(out.numpy(), expected))

    def test_broadcast_sample_weight_unknown_rank_tracing(self):
        """Test broadcast helper tracing with unknown ranks."""
        @tf.function(input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ])
        def _wrap(weight, tensor):
            """Wrap broadcast helper in tracing."""
            return lf._broadcast_sample_weight(weight, tensor)

        concrete = _wrap.get_concrete_function()
        out = concrete(
            tf.constant([1.0, 2.0], dtype=tf.float32),
            tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32),
        )
        self.assertEqual(out.shape, (2, 2))

    def test_reduce_weighted_loss_helpers(self):
        """Test weighted loss reductions."""
        losses = tf.constant([1.0, 2.0], dtype=tf.float32)
        out_none = lf._reduce_weighted_loss(losses, None, tf.keras.losses.Reduction.NONE)
        self.assertTrue(np.allclose(out_none.numpy(), losses.numpy()))
        out_sum = lf._reduce_weighted_loss(losses, None, tf.keras.losses.Reduction.SUM)
        self.assertAlmostEqual(float(out_sum.numpy()), 3.0)
        out_mean = lf._reduce_weighted_loss(losses, None, tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.assertAlmostEqual(float(out_mean.numpy()), 1.5)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
